(ns raster.compiler.core.params
  "Per-leaf semantics for parameter trees.

  `(Param T)` marks a tunable leaf — gradient target, optimizer-updated.
  `(Frozen T)` marks a leaf that participates structurally but receives no gradient.

  The structure of the tree (HMap, HVec) is general TypedClojure and carries no
  trainability semantics; that lives entirely on the leaves via these wrappers.
  Consumers (AD, optimizers, serialization) dispatch on `leaf-kind`."
  (:require [raster.compiler.core.types :as types]))

(defn param?
  "True iff annotation is (Param T)."
  [annotation]
  (and (sequential? annotation) (= 'Param (first annotation))))

(defn frozen?
  "True iff annotation is (Frozen T)."
  [annotation]
  (and (sequential? annotation) (= 'Frozen (first annotation))))

(defn leaf-kind
  "Return :param, :frozen, or :plain for an annotation."
  [annotation]
  (cond
    (param? annotation)  :param
    (frozen? annotation) :frozen
    :else                :plain))

(defn inner-type
  "Strip leaf wrappers, returning the underlying annotation. Idempotent on plain types."
  [annotation]
  (types/unwrap-leaf-wrapper annotation))
