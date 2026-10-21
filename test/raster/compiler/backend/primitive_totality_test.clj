(ns raster.compiler.backend.primitive-totality-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.intrinsics :as intrinsic]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.numerical-contract :as numerics]))

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
  (is (false? (intrinsic/primitive-fp-total? 'Math/rint [Double/TYPE])))
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

(deftest primitive-library-totality-does-not-prove-source-completion-or-accuracy
  (doseq [[operation arity] [[:sin 1] [:cos 1] [:tan 1] [:exp 1] [:log 1]
                            [:pow 2] [:sqrt 1] [:floor 1]]]
    (doseq [carrier [Float/TYPE Double/TYPE]]
      (is (intrinsic/primitive-fp-total? operation (vec (repeat arity carrier))))))
  (doseq [arguments [[Object] [Double] [:double] ['double] [Long/TYPE] [] nil
                    [Double/TYPE Double/TYPE]]]
    (is (false? (intrinsic/primitive-fp-total? :exp arguments))))
  (is (false? (intrinsic/primitive-fp-total? 'user.helpers/exp [Double/TYPE])))
  (is (= :target-library (:kernel-body-math-realization (intrinsic/descriptor :exp))))
  (is (Double/isInfinite (Math/exp Double/MAX_VALUE)))
  (is (Double/isNaN (Math/log -1.0)))
  (is (Double/isNaN (Math/sqrt -1.0)))
  (is (Double/isNaN (Math/pow -1.0 0.5)))
  (is (thrown? IllegalArgumentException (float (Math/exp Double/MAX_VALUE)))
      "a total FP producer does not discharge its final checked storage conversion"))

(deftest primitive-library-totality-cannot-certify-a-result-tag-with-invalid-operands
  (let [kernel (fn [expression]
                 (body/make {:id :math-operand-control
                             :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                             :operations [(body/->ScalarCompute (body/value 'result :double) expression)]}))
        expression (body/scalar-expression :exp :double [(body/literal 1.0 :double)]
                                           {:math-realization numerics/target-library-math})]
    (is (body/kernel-body? (kernel expression)))
    (doseq [changed [(assoc expression :arguments [(body/literal 1 :long)])
                     (assoc expression :arguments [])
                     (assoc expression :arguments ['undeclared-object])
                     (assoc-in expression [:options :math-realization :accuracy] :correctly-rounded)]]
      (is (thrown? clojure.lang.ExceptionInfo (kernel changed))))
    (doseq [[operation expected] [[:exp 1] [:+ 2]]]
      (let [failure (try (kernel (body/scalar-expression operation :double []))
                         nil
                         (catch clojure.lang.ExceptionInfo error (ex-data error)))]
        (is (= {:operation operation :expected expected :actual 0} failure)
            "shared intrinsic arity validation precedes nonempty operand analysis")))))
