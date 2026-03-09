(ns raster.types.float16
  "IEEE 754 half-precision (Float16) value type.

  Uses JDK 20+ Float.float16ToFloat / Float.floatToFloat16 for conversion.
  Arithmetic is performed by promoting to float32, computing, and converting back.
  No native JVM half-precision ALU exists, so this is for storage/bandwidth savings.

  Float16 arrays transfer as short[] to GPU memory; CUDA kernels interpret as __half.

  Usage:
    (require '[raster.types.float16 :refer [float16 ->Float16]])
    (+ (float16 1.5) (float16 2.5))  ;; => Float16(4.0)"
  (:refer-clojure :exclude [+ - * /])
  (:require [raster.core :refer [defvalue deftm defval]]
            [raster.numeric]
            [raster.types.promote :as promote :refer [register-promote-rule! register-type-tag!]]))

;; Import promotion tag value classes from raster.types.promote's namespace mappings.
(when-let [tfloat-class (get (.getMappings (the-ns 'raster.types.promote)) (symbol "TFloat"))]
  (.importClass (the-ns *ns*) ^Class tfloat-class))

(when-let [tdouble-class (get (.getMappings (the-ns 'raster.types.promote)) (symbol "TDouble"))]
  (.importClass (the-ns *ns*) ^Class tdouble-class))

;; ================================================================
;; Float16 value type — stores bits as short
;; ================================================================

(defvalue Float16 [bits :- Short])

;; ================================================================
;; Conversion helpers
;; ================================================================

(defn float16
  "Create a Float16 from any number."
  [x]
  (->Float16 (Float/floatToFloat16 (float x))))

(defn float16->float
  "Convert a Float16 to a float."
  [^Float16 x]
  (Float/float16ToFloat (.bits x)))

(defn float16->double
  "Convert a Float16 to a double."
  ^double [^Float16 x]
  (double (Float/float16ToFloat (.bits x))))

;; ================================================================
;; Arithmetic — promote to float32, compute, convert back
;; ================================================================

(deftm raster.numeric/+ [x :- Float16, y :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (clojure.core/+ (Float/float16ToFloat (.bits x))
                              (Float/float16ToFloat (.bits y))))))

(deftm raster.numeric/- [x :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (clojure.core/- (Float/float16ToFloat (.bits x))))))

(deftm raster.numeric/- [x :- Float16, y :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (clojure.core/- (Float/float16ToFloat (.bits x))
                              (Float/float16ToFloat (.bits y))))))

(deftm raster.numeric/* [x :- Float16, y :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (clojure.core/* (Float/float16ToFloat (.bits x))
                              (Float/float16ToFloat (.bits y))))))

(deftm raster.numeric// [x :- Float16, y :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (clojure.core// (Float/float16ToFloat (.bits x))
                              (Float/float16ToFloat (.bits y))))))

;; ================================================================
;; Math functions
;; ================================================================

(deftm raster.numeric/abs [x :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (Math/abs (Float/float16ToFloat (.bits x))))))

(deftm raster.numeric/sqrt [x :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16
              (float (Math/sqrt (double (Float/float16ToFloat (.bits x))))))))

(deftm raster.numeric/zero [x :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16 (float 0.0))))

(deftm raster.numeric/one [x :- Float16] :- Float16
  (->Float16 (Float/floatToFloat16 (float 1.0))))

(deftm raster.numeric/zero? [x :- Float16]
  (== (Float/float16ToFloat (.bits x)) 0.0))

(deftm raster.numeric/real-value [x :- Float16] :- Double
  (double (Float/float16ToFloat (.bits x))))

;; ================================================================
;; Promotion rules
;; ================================================================

(defval TFloat16)
(register-type-tag! Float16 (->TFloat16))

(deftm raster.types.promote/convert [t :- TFloat16, x :- Float16] :- Float16 x)
(deftm raster.types.promote/convert [t :- TFloat16, x :- Long] :- Float16
  (->Float16 (Float/floatToFloat16 (float x))))
(deftm raster.types.promote/convert [t :- TFloat16, x :- Integer] :- Float16
  (->Float16 (Float/floatToFloat16 (float x))))
(deftm raster.types.promote/convert [t :- TFloat16, x :- Number] :- Float16
  (->Float16 (Float/floatToFloat16 (.floatValue ^Number x))))

;; Promote Float16 up the chain
(deftm raster.types.promote/convert [t :- TFloat, x :- Float16] :- Float
  (Float/float16ToFloat (.bits x)))
(deftm raster.types.promote/convert [t :- TDouble, x :- Float16] :- Double
  (double (Float/float16ToFloat (.bits x))))

(register-promote-rule! Float16 Float Float)
(register-promote-rule! Float16 Double Double)
(register-promote-rule! Integer Float16 Float16)
(register-promote-rule! Long Float16 Float16)
