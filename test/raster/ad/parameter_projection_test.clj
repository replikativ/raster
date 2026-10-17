(ns raster.ad.parameter-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.ad.bind-ctx :as bind-ctx]))

(defn- collector [parameter-tag cotangent-tag]
  (let [parameter (with-meta 'parameter {:raster.type/tag parameter-tag})
        cotangent (with-meta 'cotangent {:raster.type/tag cotangent-tag})
        result (with-bindings {#'reverse/*gensym-counter* (atom 0)}
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
