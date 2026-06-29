(ns raster.compiler.core.params-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.params :as params]
            [raster.compiler.core.types :as types]))

(deftest leaf-wrappers-are-transparent-for-tags
  (testing "Param T tags identically to T"
    (is (= (types/annotation->tag '(Array double) nil)
           (types/annotation->tag '(Param (Array double)) nil)))
    (is (= (types/annotation->tag 'Double nil)
           (types/annotation->tag '(Param Double) nil))))
  (testing "Frozen T tags identically to T"
    (is (= (types/annotation->tag '(Array double) nil)
           (types/annotation->tag '(Frozen (Array double)) nil))))
  (testing "Stacked wrappers strip to inner type"
    ;; not a recommended pattern, but should not crash
    (is (= (types/annotation->tag '(Array long) nil)
           (types/annotation->tag '(Param (Frozen (Array long))) nil)))))

(deftest leaf-wrappers-are-transparent-for-hints
  (is (= (types/annotation->hint '(Array double))
         (types/annotation->hint '(Param (Array double)))))
  (is (= (types/annotation->hint '(Array float))
         (types/annotation->hint '(Frozen (Array float))))))

(deftest leaf-wrappers-are-transparent-for-record-and-prim-hints
  (is (= (types/annotation->record-hint 'Double)
         (types/annotation->record-hint '(Param Double))))
  (is (= (types/annotation->prim-hint '(Array long))
         (types/annotation->prim-hint '(Param (Array long))))))

(deftest leaf-kind-classifier
  (is (= :param  (params/leaf-kind '(Param (Array double)))))
  (is (= :frozen (params/leaf-kind '(Frozen (Array double)))))
  (is (= :plain  (params/leaf-kind '(Array double))))
  (is (= :plain  (params/leaf-kind 'Double))))

(deftest predicates
  (is (params/param? '(Param Double)))
  (is (not (params/param? '(Frozen Double))))
  (is (params/frozen? '(Frozen Double)))
  (is (not (params/frozen? '(Array double)))))

(deftest inner-type-unwraps
  (is (= '(Array double) (params/inner-type '(Param (Array double)))))
  (is (= '(Array double) (params/inner-type '(Frozen (Array double)))))
  (is (= '(Array double) (params/inner-type '(Array double))))
  (is (= 'Double (params/inner-type 'Double))))
