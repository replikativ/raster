(ns raster.compiler.passes.scalar.normalize
  "Scalar call canonicalization.

	Normalizes devirtualized scalar arithmetic .invk calls back into ordinary
	S-expression heads before simplification. Uses the :raster.op/original metadata
	attached by the walker to recover the original operator identity. Falls back to
	mangled-name pattern matching for legacy forms without metadata."
  (:require [clojure.string :as str]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.form :as form]))

(defn- invk-args
  [form]
  (vec (nnext form)))

(defn- normalizable-op?
  "True if op should be denormalized from .invk back to a direct call for simplification."
  [op]
  (or (descriptor/arithmetic-op? op) (descriptor/power-op? op)))

(defn- direct-op-for-impl
  "Legacy fallback: recover op from mangled impl name when :raster.op/original metadata is absent."
  [impl-name]
  (cond
    (str/includes? impl-name "_plus_") 'raster.numeric/+
    (str/includes? impl-name "_minus_") 'raster.numeric/-
    (str/includes? impl-name "_star_") 'raster.numeric/*
    (str/includes? impl-name "_div_") 'raster.numeric//
    (or (str/includes? impl-name "pow_m_")
        (str/includes? impl-name "pow__")) 'Math/pow
    :else nil))

(defn normalize-1
  "Normalize a single scalar call shape without recursing.
	Prefers :raster.op/original metadata from the walker; falls back to mangled-name matching."
  [form]
  (if (and (seq? form) (= '.invk (first form)))
    (if-let [op (:raster.op/original (meta form))]
      ;; Metadata path: use walker-provided original op
      (if (normalizable-op? op)
        (list* op (invk-args form))
        form)
      ;; Legacy fallback: pattern-match mangled name
      (let [impl (name (second form))]
        (if-let [op (direct-op-for-impl impl)]
          (list* op (invk-args form))
          form)))
    form))

(defn normalize
  "Recursively normalize scalar call shapes in an S-expression tree."
  [form]
  (cond
    (not (coll? form)) form

    (seq? form)
    (let [head (first form)]
      (cond
        (and (symbol? head) (#{'fn 'fn*} head))
        form

        (and (symbol? head) (contains? #{:binding :scope} (:kind (form/form-info form))))
        (let [[let-sym bindings & body] form
              normalized-bindings (vec (mapcat (fn [[sym init]]
                                                 [sym (normalize init)])
                                               (partition 2 bindings)))
              normalized-body (map normalize body)]
          (normalize-1 (let [r (apply list let-sym normalized-bindings normalized-body)]
                         (if-let [m (meta form)] (with-meta r m) r))))

        (= head 'if)
        (let [[_ test then else] form
              r (list 'if (normalize test) (normalize then) (when else (normalize else)))]
          (normalize-1 (if-let [m (meta form)] (with-meta r m) r)))

        (= head 'do)
        (let [r (apply list 'do (map normalize (rest form)))]
          (normalize-1 (if-let [m (meta form)] (with-meta r m) r)))

        :else
        (let [r (apply list (map normalize form))]
          (normalize-1 (if-let [m (meta form)] (with-meta r m) r)))))

    (vector? form) (mapv normalize form)
    (map? form) (into {} (map (fn [[k v]] [(normalize k) (normalize v)])) form)
    (set? form) (into #{} (map normalize) form)
    :else form))
