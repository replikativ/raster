(ns raster.compiler.passes.scalar.simplify
  "Pattern-based algebraic simplification.

  This is the canonical scalar simplifier namespace. It handles both raw
  arithmetic forms and iterates bottom-up to a fixed point. Typed .invk
  boundaries are retained; generic algebra may simplify their children but
  cannot erase selected operand conversions or result rounding."
  (:require [pattern :refer [rule rule-list simplifier]]
            [pattern.nanopass.dialect :refer [=> dialects]]
            [raster.compiler.ir.dialects]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.effects :as effects]))

;; ================================================================
;; Operator recognition predicates — delegate to op-descriptor
;; ================================================================

(def plus-op?  descriptor/addition-op?)
(def minus-op? descriptor/subtraction-op?)
(def mul-op?   descriptor/multiplication-op?)
(def div-op?   descriptor/division-op?)

(defn- core-operation? [op]
  ;; A generic numeric family is not an implementation witness: raster.numeric
  ;; methods can round, promote, wrap, or dispatch on user-defined carriers.
  (and (symbol? op) (= "clojure.core" (namespace op))))

(defn- core-plus? [op] (and (core-operation? op) (plus-op? op)))
(defn- core-minus? [op] (and (core-operation? op) (minus-op? op)))
(defn- core-mul? [op] (and (core-operation? op) (mul-op? op)))
(defn- core-div? [op] (and (core-operation? op) (div-op? op)))

;; ================================================================
;; Algebraic simplification rules
;; ================================================================

(def arithmetic-rules
  (rule-list
   ;; Raw symbolic operands have no retained type, finiteness, conversion or
   ;; rounding contract. Even x * 1 can change the result carrier. Keep those
   ;; operations; only fold literal core arithmetic with its actual semantics.
   (rule '((? op ~core-plus?) (? a number?) (? b number?))
         (+ a b))
   (rule '((? op ~core-minus?) (? a number?))
         (- a))
   (rule '((? op ~core-minus?) (? a number?) (? b number?))
         (- a b))
   (rule '((? op ~core-mul?) (? a number?) (? b number?))
         (* a b))
   (rule '((? op ~core-div?) (? a number?) (? b number?))
         (when (not (zero? b))
           (/ a b)))))

;; ================================================================
;; Math function rules
;; ================================================================

(def math-rules
  (rule-list
   ;; Replacing pow with another operation also changes rounding, signed-zero
   ;; behavior and the operand conversion boundary. Literal Math/pow is known.
   (rule '(Math/pow (? a number?) (? b number?))
         (Math/pow (double a) (double b)))))

(def math-const-rules
  (rule-list
   (rule '(Math/sin (? x number?)) (Math/sin (double x)))
   (rule '(Math/cos (? x number?)) (Math/cos (double x)))
   (rule '(Math/exp (? x number?)) (Math/exp (double x)))
   (rule '(Math/log (? x number?)) (when (pos? x) (Math/log (double x))))
   (rule '(Math/sqrt (? x number?)) (when (>= (double x) 0.0) (Math/sqrt (double x))))
   ;; Overloaded Math methods need an overload witness. Double literals supply
   ;; one; integer/Float/unknown calls remain for normal typed resolution.
   (rule '(Math/abs (? x double?)) (Math/abs (double x)))
   (rule '(Math/min (? a double?) (? b double?)) (Math/min (double a) (double b)))
   (rule '(Math/max (? a double?) (? b double?)) (Math/max (double a) (double b)))
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
  ;; A typed call includes operand conversion and result rounding. Generic
  ;; algebra is not authority to erase that signature-selected boundary.
  ;; Bottom-up traversal still simplifies its children.
  (if (or (not (seq? form)) (= '.invk (first form)))
    form
    (try
      (or (all-rules form) form)
      ;; A known value is not permission to move a runtime trap ahead of earlier
      ;; source effects. Leave checked overflow/division for normal evaluation.
      (catch ArithmeticException _ form))))

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
  "Strict simplification for derivative expressions, iterated to a fixpoint.
   A derivative expression is not implicitly a fast-math or real-field contract;
   symbolic zero/one terms retain their numerical and dispatch boundaries."
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

(def ^:private real-algebra-rules
  ;; Explicit formal calculus, not an executable floating-point contract. This
  ;; is used only by symbolic differentiation; compiler PE never selects it.
  (rule-list
   (rule '((? op ~plus-op?) ?x (? z number?)) (when (zero? z) x))
   (rule '((? op ~plus-op?) (? z number?) ?x) (when (zero? z) x))
   (rule '((? op ~minus-op?) ?x (? z number?)) (when (zero? z) x))
   (rule '((? op ~minus-op?) ?x ?x) (when (effects/removable-expr? x) 0))
   (rule '((? op ~minus-op?) (? a number?)) (- a))
   (rule '((? op ~mul-op?) ?x (? z number?))
         (cond (= z 1) x (and (zero? z) (effects/removable-expr? x)) 0))
   (rule '((? op ~mul-op?) (? z number?) ?x)
         (cond (= z 1) x (and (zero? z) (effects/removable-expr? x)) 0))
   (rule '((? op ~plus-op?) (? a number?) (? b number?)) (+ a b))
   (rule '((? op ~minus-op?) (? a number?) (? b number?)) (- a b))
   (rule '((? op ~mul-op?) (? a number?) (? b number?)) (* a b))
   (rule '((? op ~div-op?) (? a number?) (? b number?))
         (when (not (zero? b)) (/ a b)))))

(def ^:private real-algebra-simplify
  (simplifier
   (rule '?->form (or (real-algebra-rules form) (simplify-form form)))))

(defn simplify-symbolic-derivative
  "Normalize a formal real-algebra derivative, not an executable numeric program.
   Zero/one elimination is justified by that explicit domain, not by purity or
   by the fact that the expression happened to originate from AD."
  ([form] (simplify-symbolic-derivative form 5))
  ([form max-rounds]
   (loop [f form n 0]
     (if (>= n max-rounds)
       f
       (let [f' (real-algebra-simplify f)]
         (if (= f f') f' (recur f' (inc n))))))))
