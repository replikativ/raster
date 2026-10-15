(ns raster.ad.normalized-scalar-public-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]))

(deftm square [x :- Double] :- Double (n/* x x))
(deftm repeated-affine [x :- Double y :- Double] :- Double
  (n/+ (n/+ (n/* x x) (n/* x y)) x))
(deftm float-affine [x :- Float y :- Float] :- Float
  (n/+ (n/* x y) x))
(deftm quotient [x :- Double y :- Double] :- Double (n// x y))
(deftm float-quotient [x :- Float y :- Float] :- Float (n// x y))
(deftm subtract-aliases [x :- Double y :- Double] :- Double
  (let [product (n/* x y)
        alias product
        difference (n/- alias x)]
    (n/- difference)))

(def effect-count (atom 0))
(deftm effectful-scalar [x :- Double] :- Double
  (do (swap! effect-count inc) (n/* x x)))
(defn unsupported-scalar [x] (Math/sin (double x)))
(deftm unknown-operation [x :- Double] :- Double
  (double (unsupported-scalar x)))
(deftm narrowing-checkpoint [x :- Double] :- Double
  (double (float x)))

(deftest normalized-scalar-public-double-vertical
  (let [vg (reverse/value+grad #'square :representation :normalized-scalar)
        directional (jvp/jvp #'square :representation :normalized-scalar)]
    (doseq [x [-2.0 0.0 3.0]]
      (is (= [(* x x) (* 2.0 x)] (vg x)))
      (is (= [(* x x) (* 2.0 x 0.5)] (directional x 0.5)))
      (let [h 0.125]
        (is (= (second (vg x)) (/ (- (square (+ x h)) (square (- x h))) (* 2.0 h))))))
    (doseq [f [vg directional]]
      (is (some? (:raster.ad/normalized-bundle (meta f))))
      (is (:raster.core/deftm (meta f)))
      (is (= 1 (count (:raster.core/deftm-walked-body (meta f)))))
      (is (= (count (:raster.core/deftm-params (meta f)))
             (count (:raster.core/deftm-tags (meta f))))))
    (is (nil? (:raster.ad/normalized-bundle (meta (reverse/value+grad #'square)))))
    (is (= [9.0 6.0] ((reverse/value+grad #'square) 3.0)))))

(deftest normalized-scalar-public-exclusions-and-pairing
  (let [vg (reverse/value+grad #'repeated-affine :representation :normalized-scalar)
        selected (reverse/value+grad #'repeated-affine :representation :normalized-scalar :wrt [0])
        directional (jvp/jvp #'repeated-affine :representation :normalized-scalar)
        [value gx gy] (vg 2.0 3.0)
        [jvalue tangent] (directional 2.0 3.0 0.5 -0.25)]
    (is (= [12.0 8.0 2.0] [value gx gy]))
    (is (= [12.0 8.0 nil] (selected 2.0 3.0)))
    (is (= value jvalue))
    (is (= (+ (* gx 0.5) (* gy -0.25)) tangent))))

(deftest normalized-scalar-public-homogeneous-float
  (let [x (float 2.0) y (float 3.0)
        vg (reverse/value+grad #'float-affine :representation :normalized-scalar)
        directional (jvp/jvp #'float-affine :representation :normalized-scalar)
        result (vg x y)
        jresult (directional x y (float 0.5) (float -0.25))]
    (is (= [(float 8.0) (float 4.0) (float 2.0)] result))
    (is (every? #(instance? Float %) result))
    (is (= [(float 8.0) (float 1.5)] jresult))
    (is (every? #(instance? Float %) jresult))))

(deftest normalized-scalar-public-division
  (let [vg (reverse/value+grad #'quotient :representation :normalized-scalar)
        directional (jvp/jvp #'quotient :representation :normalized-scalar)
        [value gx gy] (vg 6.0 2.0)
        [jvalue tangent] (directional 6.0 2.0 0.5 -0.25)]
    (is (= [3.0 0.5 -1.5] [value gx gy]))
    (is (= value jvalue))
    (is (= (+ (* gx 0.5) (* gy -0.25)) tangent))
    (doseq [[slot analytic] [[0 gx] [1 gy]]]
      (let [h 1.0e-5
            args [6.0 2.0]
            plus (apply quotient (update args slot + h))
            minus (apply quotient (update args slot - h))]
        (is (< (Math/abs (- analytic (/ (- plus minus) (* 2.0 h)))) 1.0e-9)))))
  (let [result ((reverse/value+grad #'float-quotient :representation :normalized-scalar)
                (float 6.0) (float 2.0))
        directional ((jvp/jvp #'float-quotient :representation :normalized-scalar)
                     (float 6.0) (float 2.0) (float 0.5) (float -0.25))]
    (is (= [(float 3.0) (float 0.5) (float -1.5)] result))
    (is (every? #(instance? Float %) result))
    (is (= [(float 3.0) (float 0.625)] directional))
    (is (every? #(instance? Float %) directional)))
  (is (= [3.0 nil -1.5]
         ((reverse/value+grad #'quotient :representation :normalized-scalar :wrt [1]) 6.0 2.0))))

(deftest normalized-division-keeps-template-exceptional-arithmetic
  (let [vg (reverse/value+grad #'quotient :representation :normalized-scalar)
        directional (jvp/jvp #'quotient :representation :normalized-scalar)
        same-ieee? (fn [expected actual]
                     (if (Double/isNaN (double expected))
                       (Double/isNaN (double actual))
                       (= (Double/doubleToRawLongBits (double expected))
                          (Double/doubleToRawLongBits (double actual)))))]
    (doseq [[x y tx ty] [[1.0 0.0 0.0 0.0]
                        [1.0 -0.0 0.0 0.0]
                        [Double/POSITIVE_INFINITY 2.0 0.0 0.0]
                        [0.0 0.0 1.0 1.0]
                        [-0.0 2.0 1.0 0.0]]]
      ;; Independent explicit evaluation of the existing division rule, with no
      ;; reciprocal rewrite or suppression of an explicitly supplied zero seed.
      (let [value (n// x y)
            gx (n// 1.0 y)
            gy (n/- (n// (n/* 1.0 x) (n/* y y)))
            tangent (n/+ (n// tx y) (n/- (n// (n/* ty x) (n/* y y))))]
        (doseq [[expected actual] (map vector [value gx gy] (vg x y))]
          (is (same-ieee? expected actual)))
        (doseq [[expected actual] (map vector [value tangent] (directional x y tx ty))]
          (is (same-ieee? expected actual)))))))

(deftest normalized-scalar-public-subtraction-and-aliases
  (let [vg (reverse/value+grad #'subtract-aliases :representation :normalized-scalar)
        directional (jvp/jvp #'subtract-aliases :representation :normalized-scalar)]
    (is (= [-4.0 -2.0 -2.0] (vg 2.0 3.0)))
    (is (= [-4.0 -0.5] (directional 2.0 3.0 0.5 -0.25)))))

(deftest normalized-scalar-public-options-never-fall-back
  (is (= :ad-no-tangent-parameters
         (try (jvp/jvp #'square :wrt []) nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
  (is (= :ad-source-jvp-options
         (try (jvp/jvp #'square :compile? true) nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
  (doseq [construct [reverse/value+grad jvp/jvp]]
    (is (= :ad-representation
           (try (construct #'square :representation :unknown) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
    (is (= :normalized-scalar-compile-unsupported
           (try (construct #'square :representation :normalized-scalar :compile? true) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
  (is (= :normalized-scalar-mode
         (try (reverse/value+grad #'square :representation :normalized-scalar :mode :forward) nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))

(deftest normalized-scalar-public-refusals-preserve-the-module-authority
  (doseq [construct [reverse/value+grad jvp/jvp]
          [source reason] [[#'effectful-scalar :ad-normalized-operation]
                           [#'unknown-operation :ad-normalized-operation]
                           [#'narrowing-checkpoint :ad-normalized-active-conversion]]]
    (is (= reason
           (try (construct source :representation :normalized-scalar) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
  (is (zero? @effect-count) "constructing a refused derivative never executes primal effects"))
