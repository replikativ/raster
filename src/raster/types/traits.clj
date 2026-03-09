(ns raster.types.traits
  "Julia-style trait system for raster.

  Traits classify types by their capabilities, enabling trait-based
  multiple dispatch. This is the Clojure analog of Julia's Holy trait trick.

  Pattern:
    ;; 1. Query the trait for a value
    (order-style 42)        ;=> #Ordered{}
    (arithmetic-style 3.14) ;=> #FieldArithmetic{}

    ;; 2. Dispatch on trait value
    (deftm my-algo [style :- Ordered, coll :- Object] ...)
    (deftm my-algo [style :- Unordered, coll :- Object]
      (throw (ex-info \"Cannot sort unordered type\" {})))

    ;; 3. Extend for your own types
    (deftm raster.types.traits/order-style [x :- MyType] (->Ordered))"
  (:require [raster.core :refer [deftm defabstract defval]]))

;; ================================================================
;; OrderStyle — does the type support total ordering?
;; ================================================================

(defabstract OrderStyle)
(defval Ordered :implements OrderStyle)
(defval Unordered :implements OrderStyle)

(deftm order-style [x :- Long] :- Ordered (->Ordered))
(deftm order-style [x :- Double] :- Ordered (->Ordered))
(deftm order-style [x :- String] :- Ordered (->Ordered))
;; Default: unordered
(deftm order-style [x :- Object] :- Unordered (->Unordered))

;; Convenience predicates dispatching on trait values
(deftm ordered? [s :- Ordered] true)
(deftm ordered? [s :- Unordered] false)

;; ================================================================
;; ArithmeticStyle — what arithmetic does the type support?
;; ================================================================

(defabstract ArithmeticStyle)
(defval FieldArithmetic :implements ArithmeticStyle)  ;; +, -, *, /
(defval RingArithmetic :implements ArithmeticStyle)    ;; +, -, *
(defval NoArithmetic :implements ArithmeticStyle)      ;; none

(deftm arithmetic-style [x :- Double] :- FieldArithmetic (->FieldArithmetic))
(deftm arithmetic-style [x :- Long] :- RingArithmetic (->RingArithmetic))
;; Default: no arithmetic
(deftm arithmetic-style [x :- Object] :- NoArithmetic (->NoArithmetic))

;; Convenience predicates
(deftm has-division? [s :- FieldArithmetic] true)
(deftm has-division? [s :- RingArithmetic] false)
(deftm has-division? [s :- NoArithmetic] false)

(deftm has-arithmetic? [s :- FieldArithmetic] true)
(deftm has-arithmetic? [s :- RingArithmetic] true)
(deftm has-arithmetic? [s :- NoArithmetic] false)

;; ================================================================
;; IteratorSize — is the collection's length known?
;; ================================================================

(defabstract IteratorSize)
(defval HasLength :implements IteratorSize)
(defval SizeUnknown :implements IteratorSize)
(defval IsInfinite :implements IteratorSize)

(deftm iterator-size [x :- (Array double)] :- HasLength (->HasLength))
(deftm iterator-size [x :- (Array long)] :- HasLength (->HasLength))
;; Default: unknown size
(deftm iterator-size [x :- Object] :- SizeUnknown (->SizeUnknown))

;; Convenience predicates
(deftm has-length? [s :- HasLength] true)
(deftm has-length? [s :- SizeUnknown] false)
(deftm has-length? [s :- IsInfinite] false)

(deftm finite? [s :- HasLength] true)
(deftm finite? [s :- SizeUnknown] true)
(deftm finite? [s :- IsInfinite] false)
