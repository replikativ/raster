(ns raster.compiler.core.numeric-constant
  "Checked evidence for numeric literals and primitive literal casts, not general evaluation.
   Cast recognition belongs to op-descriptor. Unknown/effectful forms and failing conversions
   yield no evidence; callers must retain or decline them, never erase the conversion."
  (:require [raster.compiler.core.op-descriptor :as descriptor]))

(defn value
  "Return {:value number} for a supported constant, nil otherwise. Primitive casts use
   Clojure's checked source semantics, including floating rounding and range failures.
   No arbitrary function resolution, eval, or inference registry is involved."
  [expression]
  (cond
    (or (integer? expression) (float? expression)) {:value expression}
    (symbol? expression)
    (case expression
      Byte/MIN_VALUE {:value Byte/MIN_VALUE}
      Byte/MAX_VALUE {:value Byte/MAX_VALUE}
      Integer/MIN_VALUE {:value Integer/MIN_VALUE}
      Integer/MAX_VALUE {:value Integer/MAX_VALUE}
      Long/MIN_VALUE {:value Long/MIN_VALUE}
      Long/MAX_VALUE {:value Long/MAX_VALUE}
      Float/POSITIVE_INFINITY {:value Float/POSITIVE_INFINITY}
      Float/NEGATIVE_INFINITY {:value Float/NEGATIVE_INFINITY}
      Double/POSITIVE_INFINITY {:value Double/POSITIVE_INFINITY}
      Double/NEGATIVE_INFINITY {:value Double/NEGATIVE_INFINITY}
      nil)
    (and (seq? expression) (= 2 (count expression)))
    (when-let [tag (descriptor/cast-result-tag (first expression))]
      (when-let [operand (value (second expression))]
        (try
          {:value (case tag
                    byte (byte (:value operand))
                    int (int (:value operand))
                    long (long (:value operand))
                    float (float (:value operand))
                    double (double (:value operand)))}
          (catch IllegalArgumentException _ nil)
          (catch ArithmeticException _ nil))))
    :else nil))

(defn literal-or-original
  "Fold only a checked constant; retain every unsupported expression verbatim."
  [expression]
  (if-let [constant (value expression)] (:value constant) expression))

(defn equivalent?
  "Compare proven numeric values without rounding an integer to a float for comparison.
   NaN is never an identity. Signed-zero policy remains the algebra/schedule's responsibility."
  [left right]
  (let [l (value left) r (value right)]
    (when (and l r)
      (let [a (:value l) b (:value r)
            nonfinite? #(and (float? %) (not (Double/isFinite (double %))))
            exact-decimal #(if (float? %) (java.math.BigDecimal. (double %)) (bigdec %))]
        (if (or (nonfinite? a) (nonfinite? b))
          (and (nonfinite? a) (nonfinite? b) (== a b))
          (zero? (.compareTo ^java.math.BigDecimal (exact-decimal a) (exact-decimal b))))))))

(defn zero-value? [expression]
  (boolean (equivalent? expression 0)))
