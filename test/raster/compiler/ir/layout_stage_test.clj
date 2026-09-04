(ns raster.compiler.ir.layout-stage-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.layout-stage :as stage]))

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(deftest casts-preserve-shape-and-state-a-supported-conversion-policy
  (let [base {:id :cast :operation :cast :input 'x :output 'y
              :input-shape '[m k] :output-shape '[m k]
              :input-dtype :float :output-dtype :half
              :policy {:rounding :nearest-even :overflow :ieee}}]
    (is (stage/layout-stage? (stage/make base)))
    (is (= :layout-stage-shape
           (reason-of #(stage/make (assoc base :output-shape '[m])))))
    (is (= :layout-stage-policy
           (reason-of #(stage/make (assoc-in base [:policy :rounding] :unknown)))))
    (is (= :layout-stage-policy
           (reason-of #(stage/make (assoc-in base [:policy :overflow] :unknown)))))))

(deftest transpose-is-a-bit-preserving-rank-two-permutation
  (let [base {:id :transpose :operation :transpose :input 'x :output 'y
              :input-shape '[m k] :output-shape '[k m]
              :input-dtype :half :output-dtype :half
              :policy {:permutation [1 0]}}]
    (is (stage/layout-stage? (stage/make base)))
    (testing "shape and permutation must describe the same bijection"
      (is (= :layout-stage-shape
             (reason-of #(stage/make (assoc base :output-shape '[m k])))))
      (is (= :layout-stage-shape
             (reason-of #(stage/make (assoc base
                                            :input-shape '[b m k]
                                            :output-shape '[k m b]))))))
    (is (= :layout-stage-dtype
           (reason-of #(stage/make (assoc base :output-dtype :float)))))))
