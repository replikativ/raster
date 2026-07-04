(ns raster.ad.rrule-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Registry operations
;; ================================================================

(deftest registry-test
  (testing "Arithmetic rules registered"
    (is (tmpl/has-ad-rule? 'clojure.core/+))
    (is (tmpl/has-ad-rule? 'clojure.core/-))
    (is (tmpl/has-ad-rule? 'clojure.core/*))
    (is (tmpl/has-ad-rule? 'clojure.core//))
    (is (tmpl/has-ad-rule? '+))
    (is (tmpl/has-ad-rule? '-))
    (is (tmpl/has-ad-rule? '*))
    (is (tmpl/has-ad-rule? '/)))
  (testing "Math rules registered"
    (is (tmpl/has-ad-rule? 'Math/sin))
    (is (tmpl/has-ad-rule? 'Math/cos))
    (is (tmpl/has-ad-rule? 'Math/exp))
    (is (tmpl/has-ad-rule? 'Math/log))
    (is (tmpl/has-ad-rule? 'Math/sqrt))
    (is (tmpl/has-ad-rule? 'Math/pow))
    (is (tmpl/has-ad-rule? 'Math/abs))
    (is (tmpl/has-ad-rule? 'Math/tan))
    (is (tmpl/has-ad-rule? 'Math/asin))
    (is (tmpl/has-ad-rule? 'Math/acos))
    (is (tmpl/has-ad-rule? 'Math/atan))
    (is (tmpl/has-ad-rule? 'Math/atan2))
    (is (tmpl/has-ad-rule? 'Math/min))
    (is (tmpl/has-ad-rule? 'Math/max))
    (is (tmpl/has-ad-rule? 'Math/fma)))
  (testing "Numeric cast rules registered"
    (is (tmpl/has-ad-rule? 'double))
    (is (tmpl/has-ad-rule? 'clojure.core/double))
    (is (tmpl/has-ad-rule? 'float))
    (is (tmpl/has-ad-rule? 'long))
    (is (tmpl/has-ad-rule? 'int)))
  (testing "Unknown op not registered"
    (is (not (tmpl/has-ad-rule? 'foo/bar)))))

;; ================================================================
;; Mangled name resolution
;; ================================================================

(deftest resolve-template-test
  (testing "Direct lookup"
    (let [[r op] (tmpl/resolve-template 'Math/sin)]
      (is (some? r))
      (is (= 'Math/sin op))))
  (testing "Mangled name"
    (let [[r op] (tmpl/resolve-template '_plus__m_double_double)]
      (is (some? r))
      (is (= '+ op))))
  (testing "Mangled math"
    (let [[r op] (tmpl/resolve-template 'sin_m_double)]
      (is (some? r))
      (is (= 'Math/sin op))))
  (testing "Impl suffix"
    (let [[r op] (tmpl/resolve-template '_star__m_double_double-impl)]
      (is (some? r))
      (is (= '* op))))
  (testing "Unknown"
    (is (nil? (tmpl/resolve-template 'unknown_fn)))))
