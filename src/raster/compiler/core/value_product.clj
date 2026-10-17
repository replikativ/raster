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

(defn- constant-index [expression]
  (when (or (integer? expression)
            (and (seq? expression) (= 2 (count expression))
                 (let [operation (descriptor/semantic-op expression)]
                   (and (symbol? operation) (= "clojure.core" (namespace operation))
                        (descriptor/cast-result-tag operation)))
                 (some? (constant-index (second expression)))))
    (let [constant (numeric-constant/value expression)]
      (when (integer? (:value constant)) (:value constant)))))

(defn constant-projection
  "Describe a genuine core nth of a known-arity product, or return nil.
   Defaulted, dynamic, out-of-range, lexical or indirect calls supply no proof.
   The caller must independently establish the source product's field count."
  [expression field-count]
  (when (and (integer? field-count) (not (neg? field-count))
             (seq? expression) (= 'clojure.core/nth (first expression))
             (= 'clojure.core/nth (descriptor/semantic-op expression))
             (= 3 (count expression)))
    (when-let [index (constant-index (nth expression 2))]
      (when (<= 0 index (dec field-count))
        {:source (second expression) :index index}))))
