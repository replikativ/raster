(ns raster.compiler.ir.extent-proof-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.extent-proof :as proof]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]))

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

(deftest guarded-rectangular-products-require-nonnegative-factor-proofs
  (let [guarded '(if (< m 1) 0 (if (< n 1) 0 (* m n)))
        nonnegative (update-vals environment #(assoc % :nonnegative true))]
    (is (proof/same-volume? nonnegative guarded '[m n]))
    (is (not (proof/same-volume? environment guarded '[m n]))
        "arbitrary public Long dimensions may be negative")))

(deftest array-length-results-carry-a-nonnegative-extent-witness
  (let [equation '(= length [n]
                     (scalar {:dtypes [:long]} [input]
                             (lambda [array] (region [] [(clojure.core/alength array)]))))
        facts (dialect/default-program-facts
               {:inputs '[input]
                :values {'input (av/tensor {:dtype :float :shape ['?]})
                         'n (av/tensor {:dtype :long :shape []})}
                :equations {'length (dialect/default-equation-facts {})}})
        environment (proof/advance (proof/initial-environment
                                    (dialect/make facts [equation] '[n]))
                                   facts equation)]
    (is (true? (:nonnegative (get environment 'n))))
    (is (proof/same-volume? environment 'n '[n]))))

(deftest scalar-witnesses-become-available-only-after-their-definition
  (let [equation '(= product [p]
                     (scalar {:dtypes [:long]} [m n]
                             (lambda [x y] (region [] [(* x y)]))))
        facts (dialect/default-program-facts
               {:inputs '[m n]
                :values (zipmap '[m n p] (repeat (av/tensor {:dtype :long :shape []})))
                :equations {'product (dialect/default-equation-facts {})}})
        p (dialect/make facts [equation] '[p])
        before (proof/initial-environment p)
        after (proof/advance before facts equation)]
    (is (not (contains? before 'p)))
    (is (not (proof/same-volume? before 'p '[p])))
    (is (not (proof/same-volume? before 'p '[m n])))
    (is (proof/same-volume? after 'p '[m n]))
    (is (= after (proof/environment p)))))
