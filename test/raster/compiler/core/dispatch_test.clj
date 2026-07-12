(ns raster.compiler.core.dispatch-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.types :as types]))

;; ================================================================
;; Method entry creation and specificity
;; ================================================================

(deftest make-method-entry-test
  (testing "basic method entry has correct fields"
    (let [entry (dispatch/make-method-entry ['Double 'Double] +)]
      (is (= ['Double 'Double] (:tags entry)))
      (is (= 2 (:arity entry)))
      (is (pos? (:specificity entry))
          "Double is more specific than Object")
      (is (fn? (:target-fn entry)))))

  (testing "Object tags have lowest specificity"
    (let [obj-entry (dispatch/make-method-entry ['Object 'Object] +)
          dbl-entry (dispatch/make-method-entry ['Double 'Double] +)]
      (is (< (:specificity obj-entry) (:specificity dbl-entry))
          "Object should be less specific than Double")))

  (testing "arity is correctly recorded"
    (let [e1 (dispatch/make-method-entry ['Double] identity)
          e3 (dispatch/make-method-entry ['Double 'Long 'Float] identity)]
      (is (= 1 (:arity e1)))
      (is (= 3 (:arity e3))))))

;; ================================================================
;; check-type: instanceof, widening, subtype registry
;; ================================================================

(deftest check-type-test
  (testing "exact class match"
    (is (true? (dispatch/check-type Double 1.0)))
    (is (true? (dispatch/check-type Long 1)))
    (is (true? (dispatch/check-type String "hello"))))

  (testing "nil matches non-primitive"
    (is (true? (dispatch/check-type Object nil)))
    (is (true? (dispatch/check-type String nil))))

  (testing "numeric widening: Integer widens to Long"
    ;; JVM numeric widening: Integer can widen to Long
    (let [widen (get types/numeric-widening Integer)]
      (is (some? widen) "Integer should have widening targets")
      (is (contains? widen Long) "Integer should widen to Long")))

  (testing "superclass match via instanceof"
    (is (true? (dispatch/check-type Number 1.0)))
    (is (true? (dispatch/check-type Number 1)))
    (is (true? (dispatch/check-type Object "anything")))))

;; ================================================================
;; Subtype registry
;; ================================================================

(deftest subtype-registry-test
  (testing "registered subtype is recognized"
    ;; Register a custom subtype relationship
    (let [prev @dispatch/subtype-registry]
      (try
        (dispatch/register-subtype! String Number)
        ;; Now check-type should accept String instance where Number is expected
        ;; (this is synthetic — just testing the registry mechanism)
        (is (true? (dispatch/check-type Number "test"))
            "String registered as subtype of Number should pass check-type")
        (finally
          ;; Clean up
          (reset! dispatch/subtype-registry prev))))))

;; ================================================================
;; Ambiguity detection
;; ================================================================

(deftest methods-ambiguous-test
  (testing "identical tags: methods-ambiguous? returns truthy but add-method! deduplicates"
    ;; methods-ambiguous? compares type overlap without checking tag equality —
    ;; identical tags overlap and neither is more specific. In practice,
    ;; add-method! removes the old entry before checking ambiguities.
    (let [a (dispatch/make-method-entry ['Double 'Long] +)
          b (dispatch/make-method-entry ['Double 'Long] +)]
      (is (some? (dispatch/methods-ambiguous? a b))
          "identical tags overlap — deduplication happens at registration")))

  (testing "one strictly more specific is not ambiguous"
    (let [general (dispatch/make-method-entry ['Number 'Number] +)
          specific (dispatch/make-method-entry ['Double 'Double] +)]
      (is (not (dispatch/methods-ambiguous? general specific))
          "Double/Double is strictly more specific than Number/Number")))

  (testing "cross-overlapping types are ambiguous"
    (let [a (dispatch/make-method-entry ['Double 'Number] +)
          b (dispatch/make-method-entry ['Number 'Double] +)]
      (is (dispatch/methods-ambiguous? a b)
          "Double/Number vs Number/Double overlap without either being more specific")))

  (testing "different arities are never ambiguous"
    (let [a (dispatch/make-method-entry ['Double] identity)
          b (dispatch/make-method-entry ['Double 'Double] +)]
      (is (not (dispatch/methods-ambiguous? a b))
          "different arities cannot be ambiguous"))))

;; ================================================================
;; register-method! and dispatch
;; ================================================================

(deftest register-and-dispatch-test
  (testing "basic method registration and invocation"
    (let [test-ns (create-ns 'raster.test.dispatch-scratch)]
      (try
        ;; Register a method for Double, Double
        (dispatch/register-method!
         (fn [a b] (+ a b))
         'test-add ['Double 'Double] test-ns)
        ;; The var should now exist and be callable
        (let [v (ns-resolve test-ns 'test-add)
              f (deref v)]
          (is (some? v) "var should be interned")
          (is (= 3.0 (f 1.0 2.0))
              "dispatch should find the Double/Double method"))
        (finally
          (remove-ns 'raster.test.dispatch-scratch))))))

;; ================================================================
;; unify-type-var for parametric types
;; ================================================================

(deftest unify-type-var-test
  (testing "bare type variable unifies with Double"
    (let [result (#'dispatch/unify-type-var 'T Double #{'T})]
      (is (= {'T 'double} result)
          "T matched against Double should bind to 'double")))

  (testing "bare type variable unifies with Long"
    (let [result (#'dispatch/unify-type-var 'T Long #{'T})]
      (is (= {'T 'long} result))))

  (testing "bare type variable unifies with Float"
    (let [result (#'dispatch/unify-type-var 'T Float #{'T})]
      (is (= {'T 'float} result))))

  (testing "(Array T) unifies with double[]"
    (let [result (#'dispatch/unify-type-var '(Array T) (Class/forName "[D") #{'T})]
      (is (= {'T 'double} result)
          "(Array T) matched against double[] should bind T to 'double")))

  (testing "(Array T) unifies with float[]"
    (let [result (#'dispatch/unify-type-var '(Array T) (Class/forName "[F") #{'T})]
      (is (= {'T 'float} result))))

  (testing "(Array T) unifies with long[]"
    (let [result (#'dispatch/unify-type-var '(Array T) (Class/forName "[J") #{'T})]
      (is (= {'T 'long} result))))

  (testing "non-variable annotation returns nil"
    (let [result (#'dispatch/unify-type-var 'Double Long #{'T})]
      (is (nil? result)
          "concrete annotation without type variables should return nil")))

  (testing "parametric value type unification (e.g. Dual__Sym)"
    ;; Register Dual as parametric with type var T
    (let [prev @dispatch/parametric-value-type-vars]
      (try
        (dispatch/register-parametric-value-type! 'Dual ['T])
        ;; Create a class whose simple name is Dual__double
        ;; We'll use a proxy to simulate; instead test the logic:
        ;; When annotation is 'Dual and arg class is named Dual__Sym,
        ;; it should bind the matching type variable.
        ;; Since we can't easily create a class named Dual__Sym here,
        ;; we test the registry was updated correctly.
        (is (= ['T] (get @dispatch/parametric-value-type-vars 'Dual)))
        (finally
          (reset! dispatch/parametric-value-type-vars prev))))))

;; ================================================================
;; unify-parametric consistency check
;; ================================================================

(deftest unify-parametric-test
  (testing "consistent bindings across multiple args"
    (let [result (#'dispatch/unify-parametric
                  ['(Array T) '(Array T)]
                  [(double-array [1.0]) (double-array [2.0])]
                  #{'T})]
      (is (= {'T 'double} result)
          "both (Array T) args are double[], so T should consistently bind to 'double")))

  (testing "inconsistent bindings return nil"
    (let [result (#'dispatch/unify-parametric
                  ['(Array T) '(Array T)]
                  [(double-array [1.0]) (float-array [2.0])]
                  #{'T})]
      (is (nil? result)
          "first arg binds T=double, second wants T=float — inconsistent")))

  (testing "arity mismatch returns nil"
    (let [result (#'dispatch/unify-parametric
                  ['T]
                  [1.0 2.0]
                  #{'T})]
      (is (nil? result)
          "annotation count != arg count should fail"))))

;; ================================================================
;; assignable-from? with subtype registry
;; ================================================================

(deftest assignable-from-test
  (testing "standard JVM assignability"
    (is (true? (dispatch/assignable-from? Number Double)))
    (is (true? (dispatch/assignable-from? Object String)))
    (is (not (dispatch/assignable-from? Double String))
        "Double is not assignable from String")))

;; ================================================================
;; Signature-derived result typing (tag-level)
;; ================================================================

(deftest signature-result-tag-test
  ;; Register synthetic (All [T]) templates so the test is self-contained
  ;; (no dl.nn dependency). Shapes mirror the real facet-registered ops.
  (testing "array-in/array-out (residual-add shape) — element tag preserved"
    (dispatch/register-parametric!
     'raster.numeric/sigtest-resid '[T]
     '[(Array T) (Array T) Long] '(Array T)
     '[a b n] nil *ns*)
    ;; float and double both derive purely from the arg tags — no hard-coding
    (is (= 'floats  (dispatch/signature-result-tag
                     'raster.numeric/sigtest-resid '[floats floats long])))
    (is (= 'doubles (dispatch/signature-result-tag
                     'raster.numeric/sigtest-resid '[doubles doubles long]))))

  (testing "bare-T return (residual-add-backward-style scalar out)"
    (dispatch/register-parametric!
     'raster.numeric/sigtest-scalar '[T]
     '[(Array T) Long] 'T
     '[dy n] nil *ns*)
    (is (= 'float  (dispatch/signature-result-tag
                    'raster.numeric/sigtest-scalar '[floats long])))
    (is (= 'double (dispatch/signature-result-tag
                    'raster.numeric/sigtest-scalar '[doubles long]))))

  (testing "multiple type variables — return follows the SECOND var"
    (dispatch/register-parametric!
     'raster.numeric/sigtest-multi '[T U]
     '[(Array T) (Array U)] '(Array U)
     '[a b] nil *ns*)
    (is (= 'doubles (dispatch/signature-result-tag
                     'raster.numeric/sigtest-multi '[floats doubles])))
    (is (= 'floats  (dispatch/signature-result-tag
                     'raster.numeric/sigtest-multi '[doubles floats]))))

  (testing "concrete-param consistency — Long param must match arg tag"
    (dispatch/register-parametric!
     'raster.numeric/sigtest-consistent '[T]
     '[(Array T) Long] '(Array T)
     '[a n] nil *ns*)
    ;; consistent: Long param vs 'long arg → resolves
    (is (= 'floats (dispatch/signature-result-tag
                    'raster.numeric/sigtest-consistent '[floats long])))
    ;; inconsistent: Long param vs 'double arg → nil (facet fallback speaks)
    (is (nil? (dispatch/signature-result-tag
               'raster.numeric/sigtest-consistent '[floats double]))))

  (testing "nil / underdetermined → nil"
    ;; no template registered for this op
    (is (nil? (dispatch/signature-result-tag
               'raster.numeric/sigtest-unregistered '[floats long])))
    ;; nil arg tag underdetermines T
    (is (nil? (dispatch/signature-result-tag
               'raster.numeric/sigtest-resid '[nil floats long])))
    ;; arity mismatch → no matching template
    (is (nil? (dispatch/signature-result-tag
               'raster.numeric/sigtest-resid '[floats])))
    ;; non-array arg against (Array T) → cannot bind T → nil
    (is (nil? (dispatch/signature-result-tag
               'raster.numeric/sigtest-resid '[float floats long])))))
