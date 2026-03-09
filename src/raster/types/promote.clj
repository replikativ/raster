(ns raster.types.promote
  "Type promotion and conversion system for polymorphic arithmetic.

   Implements Julia's promote/convert pattern to eliminate N^2 cross-type
   arithmetic methods. New types only need:
   1. Same-type arithmetic: (deftm + [x :- MyType, y :- MyType] ...)
   2. Promote rule: (register-promote-rule! Long MyType MyType)
   3. Type tag + convert: (defval TMyType)
                          (register-type-tag! MyType (->TMyType))
                          (deftm convert [t :- TMyType, x :- Number] ...)

   Then mixed arithmetic (+ 1 (my-type 2)) works automatically via
   the Number+Number fallback in raster.numeric."
  (:require [raster.core :refer [deftm defval]]))

;; ================================================================
;; Type tag singletons for convert dispatch
;; ================================================================

(defval TLong)
(defval TDouble)
(defval TFloat)
(defval TInteger)
(defval TByte)
(defval TShort)

;; ================================================================
;; Registries
;; ================================================================

;; Map of [ClassA ClassB] -> ResultClass for type promotion.
(defonce ^:private promote-rules (atom {}))

;; Map of Class -> type-tag-singleton for convert dispatch.
(defonce ^:private type-tags (atom {}))

;; ================================================================
;; Registration API
;; ================================================================

(defn register-promote-rule!
  "Register that mixing type-a with type-b produces result-type.
   Symmetry is automatic — only one direction needed."
  [type-a type-b result-type]
  (swap! promote-rules assoc [type-a type-b] result-type))

(defn register-type-tag!
  "Register a type-tag singleton for a class (used by convert dispatch)."
  [cls tag]
  (swap! type-tags assoc cls tag))

(defn type-tag
  "Get the type-tag singleton for a class."
  [cls]
  (or (get @type-tags cls)
      (some (fn [[k v]] (when (.isAssignableFrom ^Class k cls) v))
            @type-tags)))

;; Register built-in tags
(register-type-tag! Long (->TLong))
(register-type-tag! Double (->TDouble))
(register-type-tag! Float (->TFloat))
(register-type-tag! Integer (->TInteger))
(register-type-tag! Byte (->TByte))
(register-type-tag! Short (->TShort))

;; ================================================================
;; Convert: extensible type conversion via deftm
;; ================================================================

;; Identity conversions
(deftm convert [t :- TLong, x :- Long] :- Long x)
(deftm convert [t :- TDouble, x :- Double] :- Double x)

;; Widening conversions
(deftm convert [t :- TDouble, x :- Long] :- Double (double x))
(deftm convert [t :- TDouble, x :- Float] :- Double (double x))
(deftm convert [t :- TDouble, x :- Number] :- Double (.doubleValue ^Number x))
(deftm convert [t :- TLong, x :- Number] :- Long (.longValue ^Number x))

;; Float conversions
(deftm convert [t :- TFloat, x :- Float] :- Float x)
(deftm convert [t :- TFloat, x :- Long] :- Float (float x))
(deftm convert [t :- TFloat, x :- Integer] :- Float (float x))
(deftm convert [t :- TFloat, x :- Number] :- Float (.floatValue ^Number x))

;; Integer conversions
(deftm convert [t :- TInteger, x :- Integer] :- Integer x)
(deftm convert [t :- TInteger, x :- Byte] :- Integer (int x))
(deftm convert [t :- TInteger, x :- Short] :- Integer (int x))
(deftm convert [t :- TInteger, x :- Number] :- Integer (.intValue ^Number x))

;; Byte/Short conversions
(deftm convert [t :- TByte, x :- Byte] :- Byte x)
(deftm convert [t :- TShort, x :- Short] :- Short x)
(deftm convert [t :- TShort, x :- Byte] :- Short (short x))

;; ================================================================
;; Promote type: determine common type for two classes
;; ================================================================

(defn promote-type
  "Given two classes, return the class they should both convert to.
   Returns nil if no promotion rule exists."
  [^Class a ^Class b]
  (if (= a b)
    a
    (or (get @promote-rules [a b])
        (get @promote-rules [b a]))))

;; ================================================================
;; Promote: convert two values to a common type
;; ================================================================

(defn promote
  "Convert two values to a common type using registered promotion rules.
   Returns [promoted-a promoted-b] or nil if no rule exists."
  [a b]
  (let [ta (class a) tb (class b)]
    (if (= ta tb)
      [a b]
      (when-let [target (promote-type ta tb)]
        (when-let [ttag (type-tag target)]
          [(convert ttag a) (convert ttag b)])))))

;; ================================================================
;; Built-in promotion rules
;; ================================================================

;; Long rules
(register-promote-rule! Long Double Double)
(register-promote-rule! Long Float Float)

;; Float rules
(register-promote-rule! Float Double Double)

;; Integer rules
(register-promote-rule! Integer Long Long)
(register-promote-rule! Integer Double Double)
(register-promote-rule! Integer Float Float)

;; Byte rules
(register-promote-rule! Byte Short Short)
(register-promote-rule! Byte Integer Integer)
(register-promote-rule! Byte Long Long)

;; Short rules
(register-promote-rule! Short Integer Integer)
(register-promote-rule! Short Long Long)
