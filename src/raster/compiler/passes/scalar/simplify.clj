(ns raster.compiler.passes.scalar.simplify
  "Pattern-based algebraic simplification.

  This is the canonical scalar simplifier namespace. It handles both raw
  arithmetic forms and devirtualized walked forms, including mangled deftm
  names and .invk calls, and iterates bottom-up to a fixed point."
  (:require [pattern :refer [rule rule-list simplifier]]
            [pattern.nanopass.dialect :refer [=> dialects]]
            [raster.compiler.ir.dialects]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.normalize :as normalize]))

;; ================================================================
;; Operator recognition predicates — delegate to op-descriptor
;; ================================================================

(def plus-op?  descriptor/addition-op?)
(def minus-op? descriptor/subtraction-op?)
(def mul-op?   descriptor/multiplication-op?)
(def div-op?   descriptor/division-op?)
(def ^:private pow-op? descriptor/power-op?)

(defn- numeric-zero? [x]
  (and (number? x) (zero? x)))

(defn- numeric-one? [x]
  (and (number? x) (== x 1)))

;; ================================================================
;; Algebraic simplification rules
;; ================================================================

(def arithmetic-rules
  (rule-list
   (rule '((? op ~plus-op?) ?x (? _ ~numeric-zero?))
         x)
   (rule '((? op ~plus-op?) (? _ ~numeric-zero?) ?x)
         x)
   (rule '((? op ~plus-op?) (? a number?) (? b number?))
         (+ a b))

   (rule '((? op ~minus-op?) ?x (? _ ~numeric-zero?))
         x)
   (rule '((? op ~minus-op?) ?x ?x)
         0.0)
   (rule '((? op ~minus-op?) (? _ ~numeric-zero?))
         0.0)
   (rule '((? op ~minus-op?) (? a number?) (? b number?))
         (- a b))

   (rule '((? op ~mul-op?) ?x (? _ ~numeric-zero?))
         0.0)
   (rule '((? op ~mul-op?) (? _ ~numeric-zero?) ?x)
         0.0)
   (rule '((? op ~mul-op?) ?x (? _ ~numeric-one?))
         x)
   (rule '((? op ~mul-op?) (? _ ~numeric-one?) ?x)
         x)
   (rule '((? op ~mul-op?) ?x)
         x)
   (rule '((? op ~mul-op?) (? a number?) (? b number?))
         (* a b))

   (rule '((? op ~div-op?) ?x (? _ ~numeric-one?))
         x)
   (rule '((? op ~div-op?) (? _ ~numeric-zero?) ?x)
         0.0)
   (rule '((? op ~div-op?) (? a number?) (? b number?))
         (when (not (zero? b))
           (/ (double a) (double b))))))

;; ================================================================
;; Math function rules
;; ================================================================

(def math-rules
  (rule-list
   (rule '((? op ~pow-op?) ?x (? _ ~numeric-zero?))
         1.0)
   (rule '((? op ~pow-op?) ?x (? _ ~numeric-one?))
         x)
   (rule '((? op ~pow-op?) ?x (? e number?))
         (when (== (double e) 2.0)
           (list 'raster.numeric/* x x)))
   (rule '((? op ~pow-op?) ?x (? e number?))
         (when (== (double e) 0.5)
           (list 'Math/sqrt x)))
   (rule '((? op ~pow-op?) ?x (? e number?))
         (when (== (double e) -1.0)
           (list 'raster.numeric// 1.0 x)))
   (rule '((? op ~pow-op?) (? a number?) (? b number?))
         (Math/pow (double a) (double b)))))

(def math-const-rules
  (rule-list
   (rule '(Math/sin (? x number?)) (Math/sin (double x)))
   (rule '(Math/cos (? x number?)) (Math/cos (double x)))
   (rule '(Math/exp (? x number?)) (Math/exp (double x)))
   (rule '(Math/log (? x number?)) (when (pos? x) (Math/log (double x))))
   (rule '(Math/sqrt (? x number?)) (when (>= (double x) 0.0) (Math/sqrt (double x))))
   (rule '(Math/abs (? x number?)) (Math/abs (double x)))
   (rule '(Math/min (? a number?) (? b number?)) (Math/min (double a) (double b)))
   (rule '(Math/max (? a number?) (? b number?)) (Math/max (double a) (double b)))
   (rule '(Math/atan2 (? a number?) (? b number?)) (Math/atan2 (double a) (double b)))))

;; ================================================================
;; Combined simplifier
;; ================================================================

(def ^:private all-rules
  (rule-list arithmetic-rules math-rules math-const-rules))

(defn- simplify-form
  "Apply simplification rules to a single form.
  Handles both direct calls and .invk calls."
  [form]
  (let [normalized (normalize/normalize-1 form)]
    (if-not (seq? normalized)
      normalized
      (or (all-rules normalized)
          normalized))))

(defn simplify-1
  "Apply one round of simplification rules to a single form.
  This preserves the old PE-facing API while keeping simplify.clj as the
  canonical simplifier namespace."
  [form]
  (simplify-form form))

(def algebraic-simplify
  "Bottom-up algebraic simplification on S-expressions.
  Applies rules to all subexpressions, iterating to fixed point.

  Handles let*/loop*/if/do forms structurally, recursing into
  subexpressions while preserving special form structure.

  Dialect: CSEEliminated => Simplified"
  (dialects (=> CSEEliminated Simplified)
            (simplifier (rule '?->form (simplify-form form)))))

(defn simplify
  "Simplify a walked S-expression using the canonical rule-based simplifier."
  [form]
  (algebraic-simplify form))

(defn simplify-derivative
  "Extended simplification for derivative expressions.
  Applies algebraic-simplify with fixpoint iteration."
  ([form] (simplify-derivative form 5))
  ([form max-rounds]
   (loop [f form
          n 0]
     (if (>= n max-rounds)
       f
       (let [f' (simplify f)]
         (if (= f f')
           f'
           (recur f' (inc n))))))))
