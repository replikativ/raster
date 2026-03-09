(ns raster.types.complex
  "Complex number type with polymorphic arithmetic.

   Extends raster.numeric/+, -, *, / with Complex dispatch,
   so any function written with polymorphic arithmetic automatically
   works with complex numbers.

   Usage:
      (require '[raster.types.complex :refer [complex re im mag angle]])
     (+ (complex 1.0 2.0) (complex 3.0 4.0)) ;=> #Complex[4.0 + 6.0i]"
  (:refer-clojure :exclude [+ - * / abs])
  (:require [raster.core :refer [deftm defvalue defval]]
            [raster.ga.core]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.types.promote :as promote]))

;; Import Clifford into this namespace by fetching the Class from raster.ga.core's
;; namespace mappings. defabstract generates interfaces via gen-interface into a
;; DynamicClassLoader, so RT/classForName can't find them cross-namespace.
(when-let [clifford-class (get (.getMappings (the-ns 'raster.ga.core)) (symbol "Clifford"))]
  (.importClass (the-ns *ns*) ^Class clifford-class))

;; ================================================================
;; Complex number type
;; ================================================================

(defvalue Complex [re :- Double, im :- Double]
  :implements [Clifford])

(defn complex
  "Create a complex number re + im*i."
  ([re] (->Complex (double re) 0.0))
  ([re im] (->Complex (double re) (double im))))

(defmethod print-method Complex [^Complex c ^java.io.Writer w]
  (.write w (str "#Complex[" (.re c)
                 (if (neg? (.im c)) " - " " + ")
                 (n/abs (.im c)) "i]")))

;; ================================================================
;; Arithmetic: Complex × Complex
;; ================================================================

(deftm raster.numeric/+ [x :- Complex, y :- Complex] :- Complex
  (->Complex (clojure.core/+ (.re x) (.re y))
             (clojure.core/+ (.im x) (.im y))))

(deftm raster.numeric/- [x :- Complex] :- Complex
  (->Complex (clojure.core/- (.re x))
             (clojure.core/- (.im x))))

(deftm raster.numeric/- [x :- Complex, y :- Complex] :- Complex
  (->Complex (clojure.core/- (.re x) (.re y))
             (clojure.core/- (.im x) (.im y))))

(deftm raster.numeric/* [x :- Complex, y :- Complex] :- Complex
  ;; (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  (let [a (.re x) b (.im x)
        c (.re y) d (.im y)]
    (->Complex (clojure.core/- (clojure.core/* a c) (clojure.core/* b d))
               (clojure.core/+ (clojure.core/* a d) (clojure.core/* b c)))))

(deftm raster.numeric// [x :- Complex, y :- Complex] :- Complex
  ;; (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
  (let [a (.re x) b (.im x)
        c (.re y) d (.im y)
        denom (clojure.core/+ (clojure.core/* c c) (clojure.core/* d d))]
    (->Complex (clojure.core// (clojure.core/+ (clojure.core/* a c) (clojure.core/* b d)) denom)
               (clojure.core// (clojure.core/- (clojure.core/* b c) (clojure.core/* a d)) denom))))

;; ================================================================
;; Arithmetic: Complex × Number and Number × Complex
;; ================================================================

(deftm raster.numeric/+ [x :- Complex, y :- Number] :- Complex
  (->Complex (clojure.core/+ (.re x) (double y)) (.im x)))

(deftm raster.numeric/+ [x :- Number, y :- Complex] :- Complex
  (->Complex (clojure.core/+ (double x) (.re y)) (.im y)))

(deftm raster.numeric/- [x :- Complex, y :- Number] :- Complex
  (->Complex (clojure.core/- (.re x) (double y)) (.im x)))

(deftm raster.numeric/- [x :- Number, y :- Complex] :- Complex
  (->Complex (clojure.core/- (double x) (.re y))
             (clojure.core/- (.im y))))

(deftm raster.numeric/* [x :- Complex, y :- Number] :- Complex
  (let [y (double y)]
    (->Complex (clojure.core/* (.re x) y)
               (clojure.core/* (.im x) y))))

(deftm raster.numeric/* [x :- Number, y :- Complex] :- Complex
  (let [x (double x)]
    (->Complex (clojure.core/* x (.re y))
               (clojure.core/* x (.im y)))))

(deftm raster.numeric// [x :- Complex, y :- Number] :- Complex
  (let [y (double y)]
    (->Complex (clojure.core// (.re x) y)
               (clojure.core// (.im x) y))))

(deftm raster.numeric// [x :- Number, y :- Complex] :- Complex
  (let [c (.re y) d (.im y)
        denom (clojure.core/+ (clojure.core/* c c) (clojure.core/* d d))
        x (double x)]
    (->Complex (clojure.core// (clojure.core/* x c) denom)
               (clojure.core// (clojure.core/- (clojure.core/* x d)) denom))))

;; ================================================================
;; Complex-specific operations
;; ================================================================

(defn re
  "Real part of a complex number."
  ^double [^Complex z] (.re z))

(defn im
  "Imaginary part of a complex number."
  ^double [^Complex z] (.im z))

(deftm mag
  "Magnitude (absolute value) of a complex number."
  [z :- Complex] :- Double
  (n/sqrt (clojure.core/+ (clojure.core/* (.re z) (.re z))
                          (clojure.core/* (.im z) (.im z)))))

(deftm raster.numeric/abs [z :- Complex] :- Double
  (n/sqrt (clojure.core/+ (clojure.core/* (.re z) (.re z))
                          (clojure.core/* (.im z) (.im z)))))

(deftm raster.numeric/zero? [z :- Complex]
  (and (== 0.0 (.re z)) (== 0.0 (.im z))))

(deftm raster.numeric/zero [z :- Complex] :- Complex
  (->Complex 0.0 0.0))

(deftm raster.numeric/one [z :- Complex] :- Complex
  (->Complex 1.0 0.0))

(deftm angle
  "Phase angle (argument) of a complex number in radians."
  [z :- Complex] :- Double
  (m/atan2 (.im z) (.re z)))

(deftm conj*
  "Complex conjugate: negate the imaginary part."
  [z :- Complex] :- Complex
  (->Complex (.re z) (clojure.core/- (.im z))))

;; ================================================================
;; Math functions for Complex
;; ================================================================

(deftm cexp
  "Complex exponential: exp(a+bi) = e^a * (cos(b) + i*sin(b))."
  [z :- Complex] :- Complex
  (let [ea (m/exp (.re z))]
    (->Complex (clojure.core/* ea (m/cos (.im z)))
               (clojure.core/* ea (m/sin (.im z))))))

(deftm csin
  "Complex sine: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)."
  [z :- Complex] :- Complex
  (let [a (.re z) b (.im z)]
    (->Complex (clojure.core/* (m/sin a) (m/cosh b))
               (clojure.core/* (m/cos a) (m/sinh b)))))

(deftm ccos
  "Complex cosine: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)."
  [z :- Complex] :- Complex
  (let [a (.re z) b (.im z)]
    (->Complex (clojure.core/* (m/cos a) (m/cosh b))
               (clojure.core/- (clojure.core/* (m/sin a) (m/sinh b))))))

(deftm clog
  "Complex natural logarithm: log(z) = ln|z| + i*arg(z)."
  [z :- Complex] :- Complex
  (->Complex (m/log (n/sqrt (clojure.core/+ (clojure.core/* (.re z) (.re z))
                                            (clojure.core/* (.im z) (.im z)))))
             (m/atan2 (.im z) (.re z))))

(deftm csqrt
  "Complex square root via polar form: sqrt(|z|) * cis(arg(z)/2)."
  [z :- Complex] :- Complex
  (let [r (n/sqrt (n/sqrt (clojure.core/+ (clojure.core/* (.re z) (.re z))
                                          (clojure.core/* (.im z) (.im z)))))
        half-angle (clojure.core/* 0.5 (m/atan2 (.im z) (.re z)))]
    (->Complex (clojure.core/* r (m/cos half-angle))
               (clojure.core/* r (m/sin half-angle)))))

;; ================================================================
;; Type promotion registration
;; ================================================================

(defval TComplex)
(promote/register-type-tag! Complex (->TComplex))
(promote/register-promote-rule! Long Complex Complex)
(promote/register-promote-rule! Double Complex Complex)

(deftm raster.types.promote/convert [t :- TComplex, x :- Complex] :- Complex x)
(deftm raster.types.promote/convert [t :- TComplex, x :- Number] :- Complex
  (->Complex (.doubleValue ^Number x) 0.0))
