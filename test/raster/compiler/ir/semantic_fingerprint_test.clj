(ns raster.compiler.ir.semantic-fingerprint-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.reference-completion-test :as reference]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]))

(defrecord Example [a b])

(deftest semantic-equality-preserves-floating-bits
  (is (fingerprint/equivalent? {:x Double/NaN} {:x Double/NaN}))
  (is (not (fingerprint/equivalent? 0.0 -0.0)))
  (is (not (fingerprint/equivalent? (int 1) (long 1))))
  (is (fingerprint/equivalent? (with-meta 'x {:tag 'long :line 1})
                              (with-meta 'x {:tag 'long :line 2})))
  (is (not (fingerprint/equivalent? (with-meta 'x {:tag 'long})
                                   (with-meta 'x {:tag 'int})))))

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
    (doseq [value [(fn [] nil) (Object.) #'fingerprint/schema
                   (.getMethod String "length" (make-array Class 0))]
            encode [fingerprint/canonical-bytes fingerprint/fingerprint
                    #(fingerprint/equivalent? % %)
                    #(fingerprint/retained-source-equivalent? % %)]]
      (is (= :semantic-fingerprint-unsupported (reason-of #(encode value)))))))

(deftest retained-source-classes-use-pair-local-identity-in-every-position
  (let [[left right] (#'reference/duplicate-interface-classes)]
    (is (= (.getName ^Class left) (.getName ^Class right)))
    (is (not (identical? left right)))
    (is (fingerprint/retained-source-equivalent? [left right left] [left right left]))
    (is (not (fingerprint/retained-source-equivalent? [left right] [right left])))
    (is (not (fingerprint/retained-source-equivalent? left right)))
    (is (fingerprint/retained-source-equivalent?
         (array-map left #{right left} right Double/TYPE)
         (array-map right Double/TYPE left #{left right})))
    (is (fingerprint/retained-source-equivalent?
         (with-meta 'x {:tag left :line 1}) (with-meta 'x {:line 9 :tag left})))
    (is (not (fingerprint/retained-source-equivalent?
              (with-meta 'x {:tag left}) (with-meta 'x {:tag right})))))
  (doseq [value [Double/TYPE [Double/TYPE] {:x Double/TYPE} #{Double/TYPE}
                 (with-meta 'x {:tag Double/TYPE})]
          encode [fingerprint/canonical-bytes fingerprint/fingerprint
                  #(fingerprint/equivalent? % %)]]
    (is (= :semantic-fingerprint-unsupported (reason-of #(encode value)))
        "portable APIs remain strict even when the process-local comparison accepts Classes")))

(deftest retained-source-comparison-preserves-the-pure-encoding-rules
  (let [nan-a (Double/longBitsToDouble 0x7ff8000000000001)
        nan-b (Double/longBitsToDouble 0x7ff8000000000002)]
    (doseq [[left right expected]
            [[nan-a nan-a true] [nan-a nan-b false] [0.0 -0.0 false]
             [(int 1) (long 1) false]
             [(with-meta 'x {:tag 'long :line 1}) (with-meta 'x {:tag 'long :line 2}) true]
             [(with-meta 'x {:tag 'long}) (with-meta 'x {:tag 'int}) false]
             [(array-map :a 1 :b #{3 2}) (array-map :b #{2 3} :a 1) true]]]
      (is (= expected (fingerprint/equivalent? left right)
             (fingerprint/retained-source-equivalent? left right))))))
