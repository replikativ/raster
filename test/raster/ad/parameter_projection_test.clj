(ns raster.ad.parameter-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.ad.tangent :as tangent]
            [raster.compiler.ad.bind-ctx :as bind-ctx]))

(deftest optional-sum-carrier-proof-does-not-assert-presence-or-guess-a-target
  (let [d (with-meta 'd {:raster.type/tag 'double})
        e (with-meta 'e {:raster.type/tag 'double})
        f (with-meta 'f {:raster.type/tag 'float})
        sum (list 'raster.ad.reverse/grad-acc d e)
        expression-tag #'tangent/expr-manifest-tag]
    (is (= 'double (expression-tag sum)))
    (is (= sum (tangent/project-expr 'double sum)))
    (let [project (eval (list 'fn '[d e] (tangent/project-expr 'double sum)))]
      (is (nil? (project nil nil)))
      (is (= 2.0 (project nil 2.0)))
      (is (= 5.0 (project 2.0 3.0))))
    (is (nil? (expression-tag (list 'raster.ad.reverse/grad-acc f f))))
    (is (nil? (expression-tag (list 'raster.ad.reverse/grad-acc d f))))
    (is (= 'float (expression-tag (float 1.0))))
    (is (= 'double (expression-tag 1.0)))
    (is (nil? (expression-tag (list 'raster.ad.reverse/grad-acc (float 1.0) (float 2.0)))))
    (is (nil? (expression-tag (list 'raster.ad.reverse/grad-acc (float 1.0) 2.0))))
    (is (nil? (expression-tag '(raster.ad.reverse/grad-acc unknown unknown))))
    (is (nil? (expression-tag (with-meta sum {:raster.op/lexical-callee true}))))
    (is (= 'double (expression-tag '(raster.ad.reverse/grad-acc a b) {'a d 'b e})))
    (is (nil? (expression-tag '(raster.ad.reverse/grad-acc a b) {'a 'b 'b 'a})))
    (with-redefs [reverse/grad-acc (fn [& _] (throw (AssertionError. "must not evaluate helper")))]
      (is (nil? (expression-tag sum))))))

(defn- collector [parameter-tag cotangent-tag]
  (let [parameter (with-meta 'parameter {:raster.type/tag parameter-tag})
        cotangent (with-meta 'cotangent {:raster.type/tag cotangent-tag})
        ;; These are tests of an internal nullable Π port, not public-gradient
        ;; materialization. The public boundary has its own zero contract.
        result (with-bindings {#'reverse/*gensym-counter* (atom 0)
                               #'reverse/*source-memory-context* {:internal-region? true}}
                 (#'reverse/collect-param-adjoints
                  [parameter] {parameter [cotangent]} (bind-ctx/make-ctx gensym) {}))]
    (eval (list 'fn [cotangent]
                (list 'let* (vec (get-in result [:rev-ctx :bindings]))
                      (first (:param-adj-syms result)))))))

(deftest double-parameter-projects-a-float-cotangent-with-no-float-inputs
  (let [project (collector 'double 'float)]
    (doseq [seed [(float Math/PI) (float 0.0) (float -0.0)
                  Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY Float/NaN]]
      (let [gradient (project seed)]
        (is (instance? Double gradient))
        (is (= (Double/doubleToLongBits (double seed))
               (Double/doubleToLongBits gradient)))))
    (is (nil? (project nil)) "absence must not become a cast exception or a zero")))

(deftest float-parameter-projects-double-after-accumulation
  (let [project (collector 'float 'double)]
    (doseq [seed [Math/PI 0.0 -0.0 Double/NaN]]
      (let [gradient (project seed)]
        (is (instance? Float gradient))
        (is (= (Float/floatToIntBits (float seed))
               (Float/floatToIntBits gradient)))))
    (is (nil? (project nil)))))

(deftest existing-source-float-projection-keeps-checked-overflow-explicit
  ;; This source path still uses the existing checked Clojure float conversion,
  ;; unlike generated strict IEEE storage conversions. Do not quietly change
  ;; that separate numeric contract while repairing the missing Double Π.
  (let [project (collector 'float 'double)]
    (doseq [seed [Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/MAX_VALUE]]
      (is (thrown? IllegalArgumentException (project seed))))))

(deftest double-to-double-is-still-an-identity-projection
  (let [project (collector 'double 'double)]
    (doseq [seed [Math/PI 0.0 -0.0 Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
      (is (= (Double/doubleToLongBits seed) (Double/doubleToLongBits (project seed)))))
    (is (nil? (project nil)))))
