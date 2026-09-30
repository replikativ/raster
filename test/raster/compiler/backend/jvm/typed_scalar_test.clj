(ns raster.compiler.backend.jvm.typed-scalar-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- reason-of [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo exception
                     (:reason (ex-data exception)))))

(deftest typed-region-executes-retained-ssa-and-dtypes
  (let [lambda
        (soac/lambda-form
         '[x n]
         [(soac/local-value 'scaled :double '(clojure.core/* x 2.0))
          (soac/local-value 'counted :double '(double n))]
         '[(clojure.core/+ scaled counted)])]
    (is (= [{:type :double :value 5.0}]
           (scalar/evaluate-region
            'raster.compiler.backend.jvm.typed-scalar-test lambda
            [{:type :double :value 1.5} {:type :long :value 2}]
            [:double])))))

(deftest typed-region-resolves-java-static-semantics-without-a-function-table
  (is (= [{:type :double :value 3.0}]
         (scalar/evaluate-region
          'raster.compiler.backend.jvm.typed-scalar-test
          (soac/lambda-form '[x] '[(Math/sqrt x)])
          [{:type :double :value 9.0}]
          [:double]))))

(deftest typed-region-fails-on-unbound-effectful-or-unrepresentable-values
  (testing "SSA closure is checked at execution as well as IR validation"
    (is (= :typed-scalar-unbound
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[missing]) [] [:double])))))
  (testing "the reference backend never executes an unproven side effect"
    (is (= :typed-scalar-effect
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[(println "no")]) [] [:double])))))
  (testing "FP16 requires an explicit representable host scalar contract"
    (is (= :typed-scalar-half
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[1.0]) [] [:half]))))))

(deftest typed-region-consumes-only-canonical-closed-numeric-constants
  (doseq [[expression dtype expected]
          [['java.lang.Float/NEGATIVE_INFINITY :float Float/NEGATIVE_INFINITY]
           ['Double/POSITIVE_INFINITY :double Double/POSITIVE_INFINITY]
           ['java.lang.Integer/MAX_VALUE :int Integer/MAX_VALUE]
           ['Long/MAX_VALUE :long Long/MAX_VALUE]]]
    (is (= [{:type dtype :value expected}]
           (scalar/evaluate-region
            'raster.compiler.backend.jvm.typed-scalar-test
            (soac/lambda-form [] [expression]) [] [dtype]))))
  (testing "lexical SSA bindings take precedence over literal spellings"
    (is (= 7.0 (scalar/evaluate-expression
                'raster.compiler.backend.jvm.typed-scalar-test
                {'Float/NEGATIVE_INFINITY 7.0} 'Float/NEGATIVE_INFINITY))))
  (testing "arbitrary fields are not resolved or reflected"
    (is (= :typed-scalar-unbound
           (reason-of #(scalar/evaluate-expression
                        'raster.compiler.backend.jvm.typed-scalar-test
                        {} 'java.lang.System/out))))))
