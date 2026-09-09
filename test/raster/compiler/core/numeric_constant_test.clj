(ns raster.compiler.core.numeric-constant-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.numeric-constant :as constant]))

(deftest primitive-casts-are-evaluated-not-stripped
  (doseq [tag '[byte int long float double clojure.core/float clojure.core/long]]
    (is (constant/zero-value? (list tag 0))))
  (is (= {:value (float 2147483647)} (constant/value '(float 2147483647))))
  (is (not (constant/equivalent? '(float 2147483647) 2147483647)))
  (is (not (constant/equivalent? (list 'double Long/MAX_VALUE) Long/MAX_VALUE)))
  (is (constant/equivalent? '(double (float 1)) 1))
  (is (= (Double/doubleToRawLongBits -0.0)
         (Double/doubleToRawLongBits (:value (constant/value '(double -0.0)))))))

(deftest no-evidence-for-unknown-or-failing-forms
  (doseq [form ['(int Double/POSITIVE_INFINITY) '(byte 256) '(unknown 0)
               '(float x) '(float 0 1) '(do (effect!) 0) 'x nil]]
    (is (nil? (constant/value form)))
    (is (= form (constant/literal-or-original form)))
    (is (not (constant/equivalent? form form)))))

(deftest explicit-wrapping-cast-is-not-a-checked-cast
  (doseq [n [0 2147483647 2147483648 4294967295 Long/MIN_VALUE Long/MAX_VALUE]]
    (is (= {:value (unchecked-int n)}
           (constant/value (list 'clojure.core/unchecked-int n)))))
  (is (nil? (constant/value '(int 2147483648))))
  (is (nil? (constant/value '(unchecked-int 2147483648)))
      "an unresolved bare call could be shadowed"))

(deftest infinities-and-nan
  (is (constant/equivalent? 'Float/POSITIVE_INFINITY Double/POSITIVE_INFINITY))
  (is (not (constant/equivalent? 'Float/POSITIVE_INFINITY 'Double/NEGATIVE_INFINITY)))
  (is (not (constant/equivalent? ##NaN ##NaN))))

(deftest only-known-static-limits-are-constants
  (is (constant/equivalent? '(int Integer/MAX_VALUE) Integer/MAX_VALUE))
  (is (constant/equivalent? 'Long/MIN_VALUE Long/MIN_VALUE))
  (is (constant/equivalent? 'Byte/MAX_VALUE Byte/MAX_VALUE))
  (is (constant/equivalent? '(float java.lang.Float/NEGATIVE_INFINITY) Float/NEGATIVE_INFINITY))
  (is (constant/equivalent? '(int java.lang.Integer/MAX_VALUE) Integer/MAX_VALUE))
  (is (nil? (constant/value 'java.lang.user/MAX_VALUE)))
  (is (nil? (constant/value 'user/MAX_VALUE)))
  (is (nil? (constant/value '(int user/MAX_VALUE)))))
