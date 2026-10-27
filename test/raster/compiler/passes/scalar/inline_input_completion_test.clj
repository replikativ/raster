(ns raster.compiler.passes.scalar.inline-input-completion-test
  "Inlining must preserve the actual typed invocation, not merely its body."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.pipeline :as pipeline]))

(deftm narrow-float [x :- Float] :- Float x)
(deftm unused-arrays [a :- (Array double) b :- (Array double)] :- Long 7)

(defmacro define-typed-callers []
  (let [operation (symbol (str (ns-name *ns*)) "narrow-float")
        implementation (symbol (str (ns-name *ns*)) "narrow-float_m_float-impl")
        selected (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'float})
        arrays-operation (symbol (str (ns-name *ns*)) "unused-arrays")
        arrays-implementation (symbol (str (ns-name *ns*)) "unused-arrays_m_doubles_doubles-impl")
        arrays-selected (entry/selected-definition arrays-operation arrays-implementation 2)
        arrays-receiver (with-meta arrays-implementation
                          {:raster.type/tag (get-in arrays-selected [:entry :typed-iface])
                           :raster.type/ret-tag 'long})]
    (list 'do
          (list 'raster.core/deftm 'narrow-caller
                '[x :- Double] ':- 'Double
                (list 'let ['result (list '.invk receiver 'x)] '(double result)))
          (list 'raster.core/deftm 'ordered-input-caller
                '[first-value :- Object events :- (Array long) second-value :- (Array double)] ':- 'Long
                (list 'let
                      ['result (list '.invk arrays-receiver 'first-value
                                     '(do (clojure.core/aset events 0
                                                            (inc (clojure.core/aget events 0)))
                                          second-value))]
                      '(long result))))))

(declare narrow-caller ordered-input-caller)
(define-typed-callers)

(defmacro define-return-identity-caller []
  (let [operation (symbol (str (ns-name *ns*)) "narrow-float")
        implementation (symbol (str (ns-name *ns*)) "narrow-float_m_float-impl")
        selected (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'float})]
    (list 'raster.core/deftm 'boxed-return-identity-caller '[x :- Float] ':- 'Boolean
          (list 'let ['result (list '.invk receiver 'x)]
                '(clojure.core/identical? result result)))))

(declare boxed-return-identity-caller)
(define-return-identity-caller)

(deftest inlined-typed-call-preserves-float-input-rounding
  ;; The caller's Double is legal for this actual Float formal: the existing
  ;; invocation emitter performs JVM d2f. Substituting the body directly loses it.
  (let [compiled (pipeline/compile-aot #'narrow-caller :simd? false)]
    (doseq [x [0.1 1.00000006 1.00000008 -1.00000008]]
      (let [expected (Double/doubleToRawLongBits (double (float x)))]
        (is (= expected (Double/doubleToRawLongBits (narrow-caller x))))
        (is (= expected (Double/doubleToRawLongBits (compiled x))))))))

(deftest first-input-check-precedes-second-input-effects-even-with-unused-formals
  (let [compiled (pipeline/compile-aot #'ordered-input-caller :simd? false)
        second-value (double-array [2.0])]
    (doseq [invoke [ordered-input-caller compiled]]
      (let [events (long-array 1)]
        (is (thrown? ClassCastException (invoke "wrong array" events second-value)))
        (is (= [0] (vec events)) "argument one is checked before evaluating argument two"))
      (let [events (long-array 1)]
        (is (= 7 (invoke nil events second-value)) "null passes the actual array CHECKCAST")
        (is (= [1] (vec events)) "the second argument is evaluated exactly once")))))

(deftest inlined-typed-call-retains-one-boxed-result-for-repeated-uses
  (let [compiled (pipeline/compile-aot #'boxed-return-identity-caller :simd? false)
        value (float 0.25)]
    (is (true? (boxed-return-identity-caller value)))
    (is (true? (compiled value))
        "the interface Object result is boxed once, not once per use")))
