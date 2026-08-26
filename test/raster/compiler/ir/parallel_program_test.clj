(ns raster.compiler.ir.parallel-program-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]))

(def ^:private reduce-source
  '(raster.par/reduce acc 0.0 i n (clojure.core/+ acc (clojure.core/aget values i))))

(defn- lowered-program []
  (:form (segop-lower/segop-lower-pass (list 'let* ['total reduce-source] 'total)
                                       {:target-device :cpu:0 :dtype :double
                                        :array-types {'values :double}})))

(deftest segops-are-first-class-typed-equations
  (let [p (lowered-program)
        equation (first (:equations p))]
    (is (program/parallel-program? p))
    (is (= :segop (:dialect p)))
    (is (= [:binding 'total] (:site equation)))
    (is (= ['n 'values] (:inputs p)))
    (is (= [[:binding 'total]] (:outputs p)))
    (is (= [reduce-source] (mapv :source (:equations p))))
    (is (every? av/abstract-value? (vals (:values p))))
    (is (= ['?] (:shape (get (:values p) 'values))))
    (is (= [] (:shape (get (:values p) [:binding 'total]))))
    (is (seq (:operations equation)))
    (testing "the source binder is ordinary Clojure data, not an IR side channel"
      (is (nil? (meta (first (second (:source p)))))))))

(deftest equation-consumption-is-invalidated-by-a-source-rewrite
  (let [p (lowered-program)]
    (is (seq (program/operations-for-binding p 'total reduce-source)))
    (is (nil? (program/operations-for-binding
               p 'total
               '(raster.par/reduce acc 1.0 i n
                                   (clojure.core/+ acc (clojure.core/aget values i))))))))

(deftest dialect-validation-is-a-full-conversion
  (let [p (lowered-program)]
    (try
      (program/validate! p (constantly false))
      (is false "an illegal remaining operation must fail validation")
      (catch clojure.lang.ExceptionInfo e
        (is (= :illegal-op-remains (:reason (ex-data e))))
        (is (= :segop (:target-dialect (ex-data e))))
        (is (= :none (:fallback (ex-data e))))))))
