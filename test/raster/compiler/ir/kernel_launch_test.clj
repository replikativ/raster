(ns raster.compiler.ir.kernel-launch-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(deftest launch-contracts-are-uniformly-one-to-three-dimensional
  (doseq [dims (range 1 4)]
    (let [wg (vec (repeat dims 8))
          groups (vec (repeat dims 4))
          spec (launch/spec {:workgroup-size wg :group-count groups})
          geometry (launch/geometry {:workgroup-size wg :group-count groups})]
      (is (= dims (launch/dimensions spec)))
      (is (= dims (launch/dimensions geometry)))
      (is (= wg (launch/static-workgroup-size spec)))))
  (testing "dimensionality is derived, never duplicated in a scalar dimensions field"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one to three dimensions"
                          (launch/spec {:workgroup-size [] :group-count []})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one to three dimensions"
                          (launch/spec {:workgroup-size [1 1 1 1]
                                        :group-count [1 1 1 1]}))))
  (testing "workgroup and grid dimensionality must agree"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dimensionality must match"
                          (launch/spec {:workgroup-size [8 8] :group-count [4]})))))

(deftest symbolic-launch-realizes-to-concrete-geometry
  (let [spec (launch/spec
              {:workgroup-size [256]
               :group-count [(launch/ceil-div 'n 256)]
               :shared-memory-bytes 1024})
        geometry (launch/realize spec {'n 513})]
    (is (launch/launch-spec? spec))
    (is (launch/launch-geometry? geometry))
    (is (= [256] (:workgroup-size geometry)))
    (is (= [3] (:group-count geometry)))
    (is (= 1024 (:shared-memory-bytes geometry))))
  (testing "runtime dimensions are explicit values, not arbitrary unchecked forms"
    (let [spec (launch/spec {:workgroup-size [8]
                             :group-count [(launch/runtime-value 'groups)]})]
      (is (= [7] (:group-count (launch/realize spec {'groups 7}))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer"
                            (launch/realize spec {})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer"
                            (launch/realize spec {'groups 1.5}))))))

(deftest symbolic-storage-expressions-use-the-same-checked-evaluator
  (is (= 17 (launch/resolve-expression {'(extent output) 17} '(extent output)))
      "stable compound extent identities are resolved as values, never evaluated as source")
  (is (= [1]
         (:group-count
          (launch/realize
           (launch/spec {:workgroup-size [32]
                         :group-count [(launch/ceil-div '(extent output) 32)]})
           {'(extent output) 17}))))
  (is (= 5 (launch/resolve-expression {'n 1025} (launch/ceil-div 'n 256))))
  (is (= 768
         (launch/resolve-expression
          {'m 3 'k 257}
          (launch/product 'm (launch/align-up (launch/ceil-div 'k 2) 256)))))
  (let [padded-extent (launch/sum 'n 1)]
    (is (= #{'n} (launch/expression-references padded-extent)))
    (is (= 1026 (launch/resolve-expression {'n 1025} padded-extent))))
  (let [tiles (launch/product (launch/ceil-div 'm 128) (launch/ceil-div 'n 128))
        requested-splits (launch/minimum (launch/ceil-div 128 tiles)
                                         (launch/floor-div 'k 1024)
                                         32)]
    (is (= #{'m 'n 'k} (launch/expression-references requested-splits)))
    (is (= 26 (launch/resolve-expression {'m 13 'n 640 'k 262144}
                                         requested-splits))))
  (let [remainder (body/expression :mod 'k 16)]
    (is (launch/expression? remainder))
    (is (= #{'k} (launch/expression-references remainder)))
    (is (= 14 (launch/resolve-expression {'k 510} remainder))))
  (is (= [3 2]
         (:group-count
          (launch/realize
           (launch/spec {:workgroup-size [256 1]
                         :group-count [(launch/ceil-div 'n 128)
                                       (launch/ceil-div 'm 128)]})
           {'m 129 'n 257}))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer"
                        (launch/resolve-expression {} (launch/ceil-div 'n 256))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be non-negative"
                        (launch/resolve-expression {} (launch/ceil-div -1 256))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be non-negative"
                        (launch/resolve-expression
                         {} (body/expression :mod -1 4)))))

(deftest launch-specialization-rebinds-explicit-expression-leaves
  (let [original (launch/spec
                  {:workgroup-size [256 1]
                   :group-count [(launch/ceil-div (launch/runtime-value 'n) 128)
                                 (launch/ceil-div 'm 64)]})
        specialized (launch/rebind-spec original {'n 256 'm 'rows})]
    (is (= #{} (launch/expression-references
                (first (:group-count specialized)))))
    (is (= #{'rows} (launch/expression-references
                     (second (:group-count specialized)))))
    (is (= [2 3]
           (:group-count (launch/realize specialized {'rows 129}))))))

(deftest concrete-geometry-rejects-invalid-backend-launches
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive integer"
                        (launch/geometry {:workgroup-size [0] :group-count [1]})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-negative"
                        (launch/geometry {:workgroup-size [8]
                                          :group-count [1]
                                          :shared-memory-bytes -1}))))

(deftest kernel-body-index-algebra-resolves-as-a-launch-dimension
  (testing "a scheduled body's extent expression evaluates with the binder's symbol values"
    (let [expression (body/expression :mul 'batch (body/index-cast 'seq-len :long :exact))
          resolve (fn [value] (get {'batch 4 'seq-len 6} value))]
      (is (= 24 (launch/resolve-expression resolve expression)))
      (is (= 3 (launch/resolve-expression
                resolve (body/expression :ceil-div 'seq-len 2))))
      (is (= 2 (launch/resolve-expression
                resolve (body/expression :floor-div 'seq-len 3))))))
  (testing "an unsupported divisor is refused rather than guessed"
    (is (thrown? clojure.lang.ExceptionInfo
                 (launch/resolve-expression (constantly 0)
                                            (body/expression :ceil-div 'n 'zero))))))

(deftest malformed-kernel-body-index-algebra-is-not-a-launch-expression
  (is (false? (launch/expression? (launch/->CeilDiv 'n 0))))
  (is (false? (launch/expression? (launch/->FloorDiv 'n -1))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"divisor must be positive"
                        (launch/resolve-expression {'n 4} (launch/->CeilDiv 'n 0))))
  (is (false? (launch/expression? (body/->IndexExpr :unknown ['n]))))
  (is (false? (launch/expression? (body/->IndexExpr :ceil-div ['n]))))
  (is (false? (launch/expression? (body/->IndexCast 'n :float :exact))))
  (is (false? (launch/expression? (body/->IndexCast 'n :long :wrap))))
  (is (launch/expression? (body/index-cast 'n :int64 :exact)))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"not an exact widening"
       (launch/validate-typed-expression!
        (body/index-cast 'n :int :exact) (constantly :long))))
  (let [expression (body/index-cast 'n :int64 :exact)]
    (is (= expression
           (launch/validate-typed-expression! expression (constantly :int))))))
