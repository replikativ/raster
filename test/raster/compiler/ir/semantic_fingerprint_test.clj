(ns raster.compiler.ir.semantic-fingerprint-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]))

(defrecord Example [a b])

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest unordered-values-have-canonical-content-order
  (let [expected "raster.semantic-fingerprint/v1:09414201d2c0c7b539b640b3638fe1e3d801ee475e88040cf87161b3e8a287b1"]
    (is (= expected (fingerprint/fingerprint {:a 1 :b #{3 2}})))
    (is (= expected (fingerprint/fingerprint (array-map :b #{2 3} :a 1)))))
  (is (not= (fingerprint/fingerprint [1 2])
            (fingerprint/fingerprint '(1 2))))
  (is (not= (fingerprint/fingerprint (int 1))
            (fingerprint/fingerprint (long 1))))
  (is (not= (fingerprint/fingerprint (->Example 1 2))
            (fingerprint/fingerprint {:a 1 :b 2}))))

(deftest fingerprints-ignore-locations-but-retain-semantic-metadata
  (let [form '(+ x 1)]
    (is (= (fingerprint/fingerprint (with-meta form {:line 10 :column 2 :file "a.clj"}))
           (fingerprint/fingerprint (with-meta form {:line 900 :column 30 :file "b.clj"}))))
    (is (not= (fingerprint/fingerprint (with-meta form {:tag 'double :line 10}))
              (fingerprint/fingerprint (with-meta form {:tag 'float :line 10}))))))

(deftest floating-point-identity-is-bit-exact
  (is (not= (fingerprint/fingerprint 0.0)
            (fingerprint/fingerprint -0.0)))
  (is (= (fingerprint/fingerprint Double/NaN)
         (fingerprint/fingerprint (Double/longBitsToDouble 0x7ff8000000000000)))))

(deftest runtime-objects-cannot-enter-semantic-identities
  (testing "identity, printing and closure internals are not compiler semantics"
    (is (= :semantic-fingerprint-unsupported
           (reason-of #(fingerprint/fingerprint (fn [] nil)))))
    (is (= :semantic-fingerprint-unsupported
           (reason-of #(fingerprint/fingerprint (Object.)))))))
