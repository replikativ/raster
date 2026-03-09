(ns raster.types.number-types
  "Julia-style abstract numeric type hierarchy.

  Defines abstract types analogous to Julia's Number hierarchy:
    Number
    ├── Real
    │   ├── AbstractFloat   (Double, Float, Float16)
    │   └── AbstractInteger
    │       └── Signed      (Long, Integer, Short, Byte)
    └── Complex

  These enable generic methods like:
    (deftm isinteger [x :- AbstractInteger] true)
    (deftm finite? [x :- AbstractInteger] true)
    (deftm nan? [x :- AbstractInteger] false)

  Usage:
    (require '[raster.types.number-types :refer [Real AbstractFloat AbstractInteger Signed]])"
  (:require [raster.core :refer [defabstract deftm]]
            [raster.compiler.core.dispatch :refer [register-subtype!]]
            [raster.types.algebraic-types :as alg]))

;; ================================================================
;; Abstract type hierarchy
;; ================================================================

(defabstract Real :extends alg/CommutativeRing)
(defabstract AbstractFloat :extends [Real alg/Field])
(defabstract AbstractInteger :extends [Real alg/EuclideanRing])
(defabstract Signed :extends AbstractInteger)

;; ================================================================
;; Register JVM types into hierarchy
;; ================================================================

;; Floats
(register-subtype! Double AbstractFloat)
(register-subtype! Float AbstractFloat)

;; Float16 (registered when available)
(try
  (when-let [f16-class (Class/forName "raster.Float16")]
    (register-subtype! f16-class AbstractFloat))
  (catch ClassNotFoundException _))

;; Signed integers
(register-subtype! Long Signed)
(register-subtype! Integer Signed)
(register-subtype! Short Signed)
(register-subtype! Byte Signed)

;; ================================================================
;; Generic numeric predicates
;; ================================================================

(deftm isinteger [x :- AbstractInteger] true)
(deftm isinteger [x :- AbstractFloat] false)

(deftm finite? [x :- AbstractInteger] true)

(deftm infinite? [x :- AbstractInteger] false)

(deftm nan? [x :- AbstractInteger] false)
