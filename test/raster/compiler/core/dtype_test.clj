(ns raster.compiler.core.dtype-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.dtype :as dtype]))

(deftest canonical-tag-reverse-projection
  (testing "declared JVM tags project through the canonical dtype facets"
    (is (= :float (dtype/dtype-for-scalar-tag 'float)))
    (is (= :int (dtype/dtype-for-array-tag 'ints))))
  (testing "an absent type fact remains absent"
    (is (nil? (dtype/dtype-for-scalar-tag nil)))
    (is (nil? (dtype/dtype-for-array-tag nil)))))
