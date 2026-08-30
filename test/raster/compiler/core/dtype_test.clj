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

(deftest jvm-array-storage-is-another-canonical-dtype-projection
  (is (= [:double :float :half :int :long :byte]
         (mapv dtype/dtype-for-jvm-array
               [(double-array 0) (float-array 0) (short-array 0)
                (int-array 0) (long-array 0) (byte-array 0)])))
  (is (nil? (dtype/dtype-for-jvm-array (object-array 0))))
  (is (nil? (dtype/dtype-for-jvm-array nil))))
