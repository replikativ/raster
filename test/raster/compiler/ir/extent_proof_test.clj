(ns raster.compiler.ir.extent-proof-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.extent-proof :as proof]))

(def environment
  {'m {:dtype :long :product {:const 1 :factors ['m]}}
   'n {:dtype :long :product {:const 1 :factors ['n]}}
   'i {:dtype :int :product {:const 1 :factors ['i]}}})

(deftest extent-equality-is-a-proof-not-cast-stripping
  (doseq [extent ['(* m n) '(* n m) '(* (long m) n) '(* m (* n 1))]]
    (is (proof/same-volume? environment extent '[m n])))
  (doseq [extent ['(* m n 2) '(* (int m) n) '(* (float m) n)
                  '(unchecked-multiply m n) '(raster.numeric/* i i)
                  '(* m unknown) '(* -1 m) '(* 9223372036854775807 2)]]
    (is (not (proof/same-volume? environment extent '[m n]))))
  (testing "opaque wrapping products cannot equal their mathematical expansion"
    (is (not (proof/same-volume? environment '(raster.numeric/* i i) '[i i]))))
  (testing "unknown/missing shapes and padding do not prove full coverage"
    (is (not (proof/same-volume? environment 1 nil)))
    (is (not (proof/same-volume? environment 9 [8])))
    (is (proof/same-volume? environment 8 [2 4]))))
