(ns raster.compiler.core.numeric-constant
  "Checked evidence for numeric literals and primitive literal casts, not general evaluation.
   Cast recognition belongs to op-descriptor. Unknown/effectful forms and failing conversions
   yield no evidence; callers must retain or decline them, never erase the conversion."
  (:require [raster.compiler.core.op-descriptor :as descriptor]))

(defn- checked-source-cast
  "Literal source carriers select primitive RT overloads, not boxed Number casts.
   Fractional values beyond integral limits can pass a boxed cast after truncation
   but fail the original primitive source cast. This helper grants no evidence
   for Object-context completions or nonliteral operands."
  [tag value]
  (case tag
    byte (cond (instance? Double value) (byte (.doubleValue ^Double value))
               (instance? Float value) (byte (.floatValue ^Float value))
               :else (byte value))
    int (cond (instance? Double value) (int (.doubleValue ^Double value))
              (instance? Float value) (int (.floatValue ^Float value))
              :else (int value))
    long (cond (instance? Double value) (long (.doubleValue ^Double value))
               (instance? Float value) (long (.floatValue ^Float value))
               :else (long value))
    float (cond (instance? Double value) (float (.doubleValue ^Double value))
                (instance? Float value) (float (.floatValue ^Float value))
                :else (float value))))

(defn- constant-value [expression source? remaining]
  (when (pos? remaining)
  (cond
    (or (integer? expression) (float? expression)) {:value expression}
    (and (not source?) (symbol? expression))
    (let [qualifier (namespace expression)
          ;; The analyzer qualifies Java classes. Normalize that spelling only; the closed
          ;; literal cases below still reject arbitrary classes/fields without resolving them.
          expression (if (and qualifier (.startsWith ^String qualifier "java.lang."))
                       (symbol (subs qualifier 10) (name expression))
                       expression)]
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
        nil))
    (and (seq? expression) (= 2 (count expression)))
    (when-let [tag (when (or (not source?)
                             (and (qualified-symbol? (first expression))
                                  (= "clojure.core" (namespace (first expression)))
                                  (not (:lexical-callee (descriptor/call-description expression)))))
                    (descriptor/cast-result-tag (first expression)))]
      (when-let [operand (constant-value (second expression) source? (dec remaining))]
        (let [operand-value (:value operand)]
          ;; `value` is deliberately a proof boundary. Never allow malformed/redefined evidence
          ;; to turn constant discovery into a host ClassCastException during diagnostic builds.
          (when (number? operand-value)
            (try
              {:value (case tag
                        byte (if source? (checked-source-cast tag operand-value) (byte operand-value))
                        int (if (= :wrap (descriptor/cast-integral-narrowing (first expression)))
                              (unchecked-int operand-value)
                              (if source? (checked-source-cast tag operand-value) (int operand-value)))
                        long (if source? (checked-source-cast tag operand-value) (long operand-value))
                        ;; A source conversion and a typed IEEE storage conversion are
                        ;; distinct contracts, even when they agree on ordinary inputs.
                        float (if (and source?
                                       (not= :ieee (descriptor/cast-floating-narrowing
                                                    (first expression))))
                                (checked-source-cast tag operand-value)
                                (.floatValue ^Number operand-value))
                        double (.doubleValue ^Number operand-value))}
              (catch IllegalArgumentException _ nil)
              (catch ArithmeticException _ nil))))))
    :else nil)))

(defn value
  "Return {:value number} for a supported canonical numeric constant, nil otherwise.
   Integral casts use Clojure's checked source semantics. Floating casts use Raster's
   canonical IEEE conversion policy, including infinity on finite overflow. This is NOT
   nonthrowing evidence for a source Clojure float call; use source-value for that contract.
   No arbitrary function resolution, eval, or inference registry is involved."
  [expression]
  (constant-value expression false 128))

(defn source-value
  "Prove a closed numeric literal/conversion under Clojure SOURCE semantics.
   Only exact qualified core cast identities and numeric literals are admitted. Bare or
   lexical callees, static-field spellings, unknown/effectful operands and failing casts
   provide no evidence. Unlike canonical IEEE value, finite Float overflow and Double
   infinities rejected by core float remain failures; unchecked-float keeps IEEE narrowing.
   This proves a total closed value,
   not permission to erase an independently retained evaluation obligation."
  [expression]
  (constant-value expression true 128))

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
