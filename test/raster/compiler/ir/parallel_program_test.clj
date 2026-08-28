(ns raster.compiler.ir.parallel-program-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
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
    (is (= {:scalar-types {'n :int}
            :array-types {'values :double}}
           (program/declared-parameter-types (assoc-in p [:values 'n :dtype] :int))))
    (is (seq (:operations equation)))
    (is (soac-dialect/program-form? (:algorithm equation)))
    (is (= (:operands equation) (:inputs (soac-dialect/facts (:algorithm equation)))))
    (is (= (:results equation) (soac-dialect/outputs (:algorithm equation))))
    (is (= '[n] (get-in (soac-dialect/facts (:algorithm equation))
                        [:values 'values :shape])))
    (is (= :typed-soac (:algorithm-dialect (first (:operations equation)))))
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

(deftest host-only-equations-require-an-explicit-empty-schedule-contract
  (let [p (lowered-program)
        equation (first (:equations p))
        host-only (assoc p :equations [(-> equation
                                           (assoc :operations [])
                                           (assoc-in [:attributes :host-only] true))])]
    (is (= host-only (program/validate! host-only (constantly false))))
    (testing "an empty schedule cannot arise accidentally"
      (try
        (program/validate! (assoc p :equations [(assoc equation :operations [])]))
        (is false "empty operations without :host-only must be rejected")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :parallel-program-operations (:reason (ex-data exception)))))))
    (testing "host-only equations cannot also claim scheduled operations"
      (try
        (program/validate! (assoc p :equations [(assoc-in equation
                                                          [:attributes :host-only] true)]))
        (is false "host-only plus scheduled operations is contradictory")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :parallel-program-host-only-operations
                 (:reason (ex-data exception)))))))))

(deftest source-shaped-bound-expressions-stay-on-the-compatibility-route
  (let [source '(raster.par/reduce acc 0.0 i (clojure.core/alength values)
                                   (+ acc (clojure.core/aget values i)))
        p (:form (segop-lower/segop-lower-pass
                  (list 'let* ['total source] 'total)
                  {:target-device :cpu:0 :dtype :double
                   :array-types {'values :double}}))
        equation (first (:equations p))]
    (is (nil? (:algorithm equation)))
    (is (seq (:operations equation)))
    (is (= [:binding 'total] (:site equation)))))
