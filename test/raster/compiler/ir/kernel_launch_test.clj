(ns raster.compiler.ir.kernel-launch-test
  (:require [clojure.test :refer [deftest is testing]]
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
  (is (= 5 (launch/resolve-expression {'n 1025} (launch/ceil-div 'n 256))))
  (is (= 768
         (launch/resolve-expression
          {'m 3 'k 257}
          (launch/product 'm (launch/align-up (launch/ceil-div 'k 2) 256)))))
  (let [tiles (launch/product (launch/ceil-div 'm 128) (launch/ceil-div 'n 128))
        requested-splits (launch/minimum (launch/ceil-div 128 tiles)
                                         (launch/floor-div 'k 1024)
                                         32)]
    (is (= #{'m 'n 'k} (launch/expression-references requested-splits)))
    (is (= 26 (launch/resolve-expression {'m 13 'n 640 'k 262144}
                                         requested-splits))))
  (is (= [3 2]
         (:group-count
          (launch/realize
           (launch/spec {:workgroup-size [256 1]
                         :group-count [(launch/ceil-div 'n 128)
                                       (launch/ceil-div 'm 128)]})
           {'m 129 'n 257}))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer"
                        (launch/resolve-expression {} (launch/ceil-div 'n 256)))))

(deftest concrete-geometry-rejects-invalid-backend-launches
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive integer"
                        (launch/geometry {:workgroup-size [0] :group-count [1]})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-negative"
                        (launch/geometry {:workgroup-size [8]
                                          :group-count [1]
                                          :shared-memory-bytes -1}))))
