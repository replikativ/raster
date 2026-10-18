(ns raster.compiler.backend.primitive-totality-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.intrinsics :as intrinsic]))

(deftest primitive-fp-domain-is-separate-from-types-and-operand-completion
  (doseq [operation '[clojure.core/+ clojure.core/- clojure.core/* clojure.core//
                      clojure.core/< clojure.core/> clojure.core/<= clojure.core/>=]]
    (doseq [carriers [[Double/TYPE Double/TYPE] [Float/TYPE Float/TYPE]
                     [Float/TYPE Double/TYPE]]]
      (is (intrinsic/primitive-fp-total? operation carriers)))
    (doseq [carriers [[Long/TYPE Long/TYPE] [Object Object] [Double Double]
                     [:double :double] ['double 'double] [Double/TYPE] nil]]
      (is (false? (intrinsic/primitive-fp-total? operation carriers)))))
  (doseq [operation '[user.helpers// user.helpers/_plus__m_double_double-impl
                      clojure.core/quot clojure.core/unchecked-add Math/sqrt]]
    (is (false? (intrinsic/primitive-fp-total? operation [Long/TYPE Long/TYPE]))))
  (is (false? (intrinsic/primitive-fp-total? 'user.helpers/+ [Double/TYPE Double/TYPE])))
  (is (false? (intrinsic/primitive-fp-total? 'Math/sqrt [Double/TYPE])))
  (is (thrown? ArithmeticException (apply / [1 0])))
  (is (thrown? ArithmeticException
               (apply / [(Double/valueOf 1.0) (Double/valueOf 0.0)]))
      "generic boxed floating division traps even though primitive double division is IEEE")
  (is (thrown? ArithmeticException (Math/addExact Long/MAX_VALUE 1)))
  (is (thrown? NullPointerException (double nil)))
  (is (false? (intrinsic/primitive-fp-total? 'clojure.core// [Object Object]))
      "Object invocation/unboxing is not certified by the primitive operation facet"))

(deftest ieee-edge-values-do-not-turn-primitive-arithmetic-into-exceptions
  (let [operations [(fn [^double x ^double y] (+ x y))
                    (fn [^double x ^double y] (- x y))
                    (fn [^double x ^double y] (* x y))
                    (fn [^double x ^double y] (/ x y))
                    (fn [^double x ^double y] (< x y))
                    (fn [^double x ^double y] (> x y))
                    (fn [^double x ^double y] (<= x y))
                    (fn [^double x ^double y] (>= x y))]]
    (doseq [operation operations
            [x y] [[1.0 0.0] [1.0 -0.0] [0.0 0.0]
                   [Double/MAX_VALUE Double/MAX_VALUE]
                   [Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]
                   [Double/NaN 1.0]]]
      (is (some? (operation x y)))))
  (is (= Double/POSITIVE_INFINITY (/ (double 1) (double 0))))
  (is (= Double/NEGATIVE_INFINITY (/ (double 1) (double -0.0))))
  (is (Double/isNaN (/ (double 0) (double 0)))))
