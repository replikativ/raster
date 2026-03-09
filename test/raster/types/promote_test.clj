(ns raster.types.promote-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.types.promote :as promote :refer [promote promote-type
                                                      convert register-promote-rule!
                                                      register-type-tag!]]
            [raster.numeric :refer [+ - * / zero one]]
            [raster.core :refer [deftm defvalue defval]]
            [raster.types.complex :refer [complex ->Complex re im]]
            [raster.ad.forward :refer [dual ->Dual derivative]]))

;; ================================================================
;; promote-type tests
;; ================================================================

(deftest promote-type-test
  (testing "same type returns itself"
    (is (= Long (promote-type Long Long)))
    (is (= Double (promote-type Double Double))))
  (testing "Long + Double -> Double"
    (is (= Double (promote-type Long Double)))
    (is (= Double (promote-type Double Long))))
  (testing "Long + Complex -> Complex"
    (is (= raster.types.complex.Complex (promote-type Long raster.types.complex.Complex)))
    (is (= raster.types.complex.Complex (promote-type raster.types.complex.Complex Long))))
  (testing "Double + Complex -> Complex"
    (is (= raster.types.complex.Complex (promote-type Double raster.types.complex.Complex))))
  (testing "Long + Dual -> Dual"
    (is (= raster.ad.forward.Dual (promote-type Long raster.ad.forward.Dual)))
    (is (= raster.ad.forward.Dual (promote-type raster.ad.forward.Dual Long))))
  (testing "no rule returns nil"
    (is (nil? (promote-type String Long)))))

;; ================================================================
;; convert tests
;; ================================================================

(deftest convert-test
  (testing "identity conversions"
    (is (= 42 (convert (promote/type-tag Long) 42)))
    (is (= 3.14 (convert (promote/type-tag Double) 3.14))))
  (testing "Long -> Double"
    (is (= 42.0 (convert (promote/type-tag Double) 42)))
    (is (instance? Double (convert (promote/type-tag Double) 42))))
  (testing "Long -> Complex"
    (let [c (convert (promote/type-tag raster.types.complex.Complex) 42)]
      (is (instance? raster.types.complex.Complex c))
      (is (== 42.0 (re c)))
      (is (== 0.0 (im c)))))
  (testing "Long -> Dual"
    (let [d (convert (promote/type-tag raster.ad.forward.Dual) 42)]
      (is (instance? raster.ad.forward.Dual d))
      (is (== 42.0 (.v ^raster.ad.forward.Dual d)))
      (is (== 0.0 (aget (.partials ^raster.ad.forward.Dual d) 0))))))

;; ================================================================
;; promote tests
;; ================================================================

(deftest promote-test
  (testing "same type returns pair"
    (is (= [1 2] (promote 1 2)))
    (is (= [1.0 2.0] (promote 1.0 2.0))))
  (testing "Long + Double -> [Double Double]"
    (let [[a b] (promote 1 2.0)]
      (is (== 1.0 a))
      (is (== 2.0 b))
      (is (instance? Double a))
      (is (instance? Double b))))
  (testing "Long + Complex -> [Complex Complex]"
    (let [[a b] (promote 1 (complex 2.0 3.0))]
      (is (instance? raster.types.complex.Complex a))
      (is (instance? raster.types.complex.Complex b))
      (is (== 1.0 (re a)))
      (is (== 0.0 (im a)))))
  (testing "no rule returns nil"
    (is (nil? (promote "hello" 42)))))

;; ================================================================
;; Arithmetic via promotion (cross-type without explicit methods)
;; ================================================================

(deftest arithmetic-via-promotion-test
  (testing "existing specific methods still work"
    (is (= 3 (+ 1 2)))
    (is (= 3.0 (+ 1.0 2.0)))
    (is (= 3.0 (+ 1 2.0))))
  (testing "Complex arithmetic via explicit methods"
    (let [c (+ (complex 1.0 2.0) (complex 3.0 4.0))]
      (is (== 4.0 (re c)))
      (is (== 6.0 (im c)))))
  (testing "Dual arithmetic via explicit methods"
    (let [d (+ (dual 1.0 1.0) (dual 2.0 0.0))]
      (is (== 3.0 (.v ^raster.ad.forward.Dual d)))
      (is (== 1.0 (aget (.partials ^raster.ad.forward.Dual d) 0))))))

;; ================================================================
;; User-defined type with promotion
;; ================================================================

(defvalue Rational [num :- Long, den :- Long])

(defval TRational)
(register-type-tag! Rational (->TRational))
(register-promote-rule! Long Rational Rational)

(deftm raster.types.promote/convert [t :- TRational, x :- Rational] :- Rational x)
(deftm raster.types.promote/convert [t :- TRational, x :- Long] :- Rational
  (->Rational x 1))

;; Same-type arithmetic (must use fully-qualified names to add to numeric's dispatch table)
(deftm raster.numeric/+ [x :- Rational, y :- Rational] :- Rational
  (->Rational (clojure.core/+ (clojure.core/* (.num x) (.den y))
                              (clojure.core/* (.num y) (.den x)))
              (clojure.core/* (.den x) (.den y))))

(deftm raster.numeric/* [x :- Rational, y :- Rational] :- Rational
  (->Rational (clojure.core/* (.num x) (.num y))
              (clojure.core/* (.den x) (.den y))))

(deftest user-type-promotion-test
  (testing "Long + Rational via promote (no explicit Long+Rational method)"
    (let [r (+ 1 (->Rational 1 2))]
      (is (instance? Rational r))
      ;; 1/1 + 1/2 = (1*2 + 1*1) / (1*2) = 3/2
      (is (= 3 (.num ^Rational r)))
      (is (= 2 (.den ^Rational r)))))
  (testing "Rational + Long via promote (symmetry)"
    (let [r (+ (->Rational 1 3) 2)]
      (is (instance? Rational r))
      ;; 1/3 + 2/1 = (1*1 + 2*3) / (3*1) = 7/3
      (is (= 7 (.num ^Rational r)))
      (is (= 3 (.den ^Rational r)))))
  (testing "Rational * Long via promote"
    (let [r (* 3 (->Rational 2 5))]
      (is (instance? Rational r))
      ;; 3/1 * 2/5 = 6/5
      (is (= 6 (.num ^Rational r)))
      (is (= 5 (.den ^Rational r))))))
