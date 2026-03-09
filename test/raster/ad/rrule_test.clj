(ns raster.ad.rrule-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.templates :as tmpl]
            [raster.ad.pullbacks]))

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
;; Pullback correctness
;; ================================================================

(defn- approx= [a b & {:keys [eps] :or {eps 1e-10}}]
  (< (Math/abs (- (double a) (double b))) eps))

(deftest addition-pullback-test
  (let [rule (tmpl/get-pullback-factory 'clojure.core/+)
        pb (rule 7.0 3.0 4.0)]
    (is (= [1.0 1.0] (pb 1.0)))
    (is (= [5.0 5.0] (pb 5.0)))))

(deftest subtraction-pullback-test
  (testing "Binary subtraction"
    (let [rule (tmpl/get-pullback-factory 'clojure.core/-)
          pb (rule 1.0 4.0 3.0)]
      (is (= [1.0 -1.0] (pb 1.0)))))
  (testing "Unary negation"
    (let [rule (tmpl/get-pullback-factory 'clojure.core/-)
          pb (rule -3.0 3.0)]
      (is (= [-1.0] (pb 1.0))))))

(deftest multiplication-pullback-test
  (let [rule (tmpl/get-pullback-factory 'clojure.core/*)
        pb (rule 12.0 3.0 4.0)]
    (is (= [4.0 3.0] (pb 1.0)))
    (is (= [8.0 6.0] (pb 2.0)))))

(deftest division-pullback-test
  (let [rule (tmpl/get-pullback-factory 'clojure.core//)
        pb (rule 2.0 6.0 3.0)
        [dx dy] (pb 1.0)]
    (is (approx= (/ 1.0 3.0) dx))
    (is (approx= (/ -6.0 9.0) dy))))

(deftest sin-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/sin)
        pb (rule (Math/sin 1.0) 1.0)
        [dx] (pb 1.0)]
    (is (approx= (Math/cos 1.0) dx))))

(deftest cos-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/cos)
        pb (rule (Math/cos 1.0) 1.0)
        [dx] (pb 1.0)]
    (is (approx= (- (Math/sin 1.0)) dx))))

(deftest exp-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/exp)
        result (Math/exp 2.0)
        pb (rule result 2.0)
        [dx] (pb 1.0)]
    (is (approx= result dx))))

(deftest log-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/log)
        pb (rule (Math/log 2.0) 2.0)
        [dx] (pb 1.0)]
    (is (approx= 0.5 dx))))

(deftest sqrt-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/sqrt)
        result (Math/sqrt 4.0)
        pb (rule result 4.0)
        [dx] (pb 1.0)]
    (is (approx= 0.25 dx))))

(deftest pow-pullback-test
  (let [rule (tmpl/get-pullback-factory 'Math/pow)
        result (Math/pow 2.0 3.0)
        pb (rule result 2.0 3.0)
        [dx dn] (pb 1.0)]
    (is (approx= 12.0 dx))
    (is (approx= (* 8.0 (Math/log 2.0)) dn))))

(deftest double-cast-pullback-test
  (testing "double cast passes through adjoint"
    (let [rule (tmpl/get-pullback-factory 'double)
          pb (rule 3.0 3.0)
          [dx] (pb 5.0)]
      (is (= 5.0 dx)))))

(deftest long-cast-pullback-test
  (testing "long cast has zero gradient"
    (let [rule (tmpl/get-pullback-factory 'long)
          pb (rule 3 3.0)
          [dx] (pb 1.0)]
      (is (= 0.0 dx)))))

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
