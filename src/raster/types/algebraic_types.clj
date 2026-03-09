(ns raster.types.algebraic-types
  "Default algebraic and categorical type lattice.

  This namespace provides a broad, composable hierarchy for:
  - algebraic structures (semigroup, monoid, group, ring, field)
  - linear structures (module, vector space, algebra)
  - category-theoretic structures (category-like, monoidal, traced)

  The hierarchy is intended as a default dispatch vocabulary for Raster and
  future Catlab/GATlab ports."
  (:require [raster.core :refer [defabstract deftm]]
            [raster.compiler.core.dispatch :refer [register-subtype!]]))

;; ================================================================
;; Algebraic hierarchy (carrier + operations)
;; ================================================================

(defabstract Magma)
(defabstract Semigroup :extends Magma)
(defabstract Monoid :extends Semigroup)
(defabstract Group :extends Monoid)
(defabstract AbelianGroup :extends Group)

(defabstract Semiring)
(defabstract Ring :extends [Semiring AbelianGroup])
(defabstract CommutativeRing :extends Ring)
(defabstract EuclideanRing :extends CommutativeRing)
(defabstract Field :extends EuclideanRing)

(defabstract Module)
(defabstract VectorSpace :extends Module)
(defabstract Algebra :extends [Module Ring])
(defabstract GradedAlgebra :extends Algebra)
(defabstract LieAlgebra :extends Module)

;; ================================================================
;; Category-theoretic hierarchy
;; ================================================================

(defabstract CategoryLike)
(defabstract MonoidalCategory :extends CategoryLike)
(defabstract SymmetricMonoidalCategory :extends MonoidalCategory)
(defabstract CartesianCategory :extends SymmetricMonoidalCategory)
(defabstract TracedMonoidalCategory :extends MonoidalCategory)
(defabstract CompactClosedCategory :extends SymmetricMonoidalCategory)
(defabstract Bicategory :extends CategoryLike)

;; ================================================================
;; JVM and Clojure numeric registrations
;; ================================================================

;; Integer-like rings
(register-subtype! Long EuclideanRing)
(register-subtype! Integer EuclideanRing)
(register-subtype! Short EuclideanRing)
(register-subtype! Byte EuclideanRing)
(register-subtype! clojure.lang.BigInt EuclideanRing)
(register-subtype! java.math.BigInteger EuclideanRing)

;; Rational/float-like fields
(register-subtype! clojure.lang.Ratio Field)
(register-subtype! Double Field)
(register-subtype! Float Field)

;; Decimal arithmetic is a commutative ring, but not a field in general.
(register-subtype! java.math.BigDecimal CommutativeRing)

;; Complex may be absent on some classpaths; register opportunistically.
(try
  (register-subtype! (Class/forName "raster.types.complex.Complex") Field)
  (catch ClassNotFoundException _ nil))

;; ================================================================
;; Capability predicates
;; ================================================================

(deftm has-associative-op? [_ :- Semigroup] true)
(deftm has-associative-op? [_ :- Object] false)

(deftm has-identity? [_ :- Monoid] true)
(deftm has-identity? [_ :- Object] false)

(deftm has-additive-inverse? [_ :- Group] true)
(deftm has-additive-inverse? [_ :- Object] false)

(deftm additive-commutative? [_ :- AbelianGroup] true)
(deftm additive-commutative? [_ :- Object] false)

(deftm multiplicative-commutative? [_ :- CommutativeRing] true)
(deftm multiplicative-commutative? [_ :- Object] false)

(deftm has-multiplicative-inverse? [_ :- Field] true)
(deftm has-multiplicative-inverse? [_ :- Object] false)

(deftm has-euclidean-division? [_ :- EuclideanRing] true)
(deftm has-euclidean-division? [_ :- Object] false)

(deftm category-like? [_ :- CategoryLike] true)
(deftm category-like? [_ :- Object] false)

(deftm monoidal? [_ :- MonoidalCategory] true)
(deftm monoidal? [_ :- Object] false)

(deftm symmetric-monoidal? [_ :- SymmetricMonoidalCategory] true)
(deftm symmetric-monoidal? [_ :- Object] false)

(deftm compact-closed? [_ :- CompactClosedCategory] true)
(deftm compact-closed? [_ :- Object] false)
