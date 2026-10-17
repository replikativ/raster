(ns raster.compiler.core.value-product
  "Shared syntax evidence for immutable value products and constant projections.

   These facts neither evaluate source nor infer field types, tangent spaces,
   ownership or freshness. An immutable product may contain borrowed mutable
   arrays. Consumers must retain constructor evaluation, field order, identity
   where observed, and all unsupported projection traps."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.numeric-constant :as numeric-constant]))

(defn literal-fields
  "The original fields of a literal product, preserving metadata and order.
   nil means no constructor evidence, not an absent field or an empty product."
  [expression]
  (when (vector? expression) expression))

(defn index-value
  "Prove an integer literal/cast index, optionally using constants already proved by the
   enclosing lexical owner. The lookup supplies frozen integer values, not runtime inputs,
   type declarations or unevaluated definitions that a later rebinding could change."
  ([expression] (index-value expression (constantly nil)))
  ([expression known-index]
   (cond
     (integer? expression) expression
     (symbol? expression) (let [value (known-index expression)] (when (integer? value) value))
     (and (seq? expression) (= 2 (count expression))
          (let [operation (descriptor/semantic-op expression)]
            (and (symbol? operation) (= "clojure.core" (namespace operation))
                 (descriptor/cast-result-tag operation))))
     (when-let [operand (index-value (second expression) known-index)]
       (let [constant (numeric-constant/value (list (first expression) operand))]
         (when (integer? (:value constant)) (:value constant))))
     :else nil)))

(defn constant-projection
  "Describe a genuine core nth of a known-arity product, or return nil.
   Defaulted, runtime-dynamic, out-of-range, lexical or indirect calls supply no proof.
   The caller establishes the product's arity and any frozen lexical index values."
  ([expression field-count]
   (constant-projection expression field-count (constantly nil)))
  ([expression field-count known-index]
  (when (and (integer? field-count) (not (neg? field-count))
             (seq? expression) (= 'clojure.core/nth (first expression))
             (= 'clojure.core/nth (descriptor/semantic-op expression))
             (= 3 (count expression)))
    (when-let [index (index-value (nth expression 2) known-index)]
      (when (<= 0 index (dec field-count))
        {:source (second expression) :index index})))))
