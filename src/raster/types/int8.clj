(ns raster.types.int8
  "Signed 8-bit integer (Int8) value type — the element type of quantized tensors.

  DESIGN (see memory type_semantics_vs_target_schedule): unlike Float16 — which
  promote-compute-demotes because the JVM has no half ALU — Int8's *defining*
  operation is WIDENING. int8 embeds exactly into int32/int64 and the arithmetic
  that matters (the quantized dot's accumulator) happens in the wider type and is
  never narrowed back. So we register widening promotion rules (Int8 mixed with a
  wider type → the wider type) and DO NOT provide same-width int8+int8→int8
  arithmetic — it's deferred, needs an overflow policy, and is irrelevant to the
  quant path. Storage is byte[]; (Array Int8) is a byte[] under SoA.

  The widening dot is exact integer arithmetic on every backend (int32 accumulate),
  so unlike Float16 there is no per-target fidelity divergence — Int8 is the clean
  case for `type = semantics, realization = per-target schedule`.

  Usage:
    (require '[raster.types.int8 :refer [int8 int8->long ->Int8]])
    (int8->long (int8 -7))   ;; => -7  (sign-extending widen)"
  (:require [raster.core :refer [defvalue deftm defval]]
            [raster.numeric]
            [raster.arrays]
            [raster.types.promote :as promote :refer [register-promote-rule! register-type-tag!]]))

;; Import the integer promotion tags from raster.types.promote's namespace.
(when-let [c (get (.getMappings (the-ns 'raster.types.promote)) (symbol "TInteger"))]
  (.importClass (the-ns *ns*) ^Class c))
(when-let [c (get (.getMappings (the-ns 'raster.types.promote)) (symbol "TLong"))]
  (.importClass (the-ns *ns*) ^Class c))

;; ================================================================
;; Int8 value type — stores the signed byte
;; ================================================================

(defvalue Int8 [val :- Byte])

;; ================================================================
;; Construction + WIDENING conversions (the defining operations)
;; ================================================================

(deftm int8
  "Create an Int8 from a long, truncated to signed 8-bit (wraps, like Julia)."
  [x :- Long] :- Int8
  (->Int8 (unchecked-byte x)))

(deftm int8 [x :- Integer] :- Int8
  (->Int8 (unchecked-byte (long x))))

(deftm int8
  "From a byte (e.g. a byte-array read) — already 8-bit, wrap directly."
  [x :- Byte] :- Int8
  (->Int8 x))

(deftm int8->long
  "Widen an Int8 to a (sign-extended) long — the exact embedding int8 ↪ int64."
  [x :- Int8] :- Long
  (long (.val x)))

(deftm int8->int
  "Widen an Int8 to a (sign-extended) int (returned as long — JVM int-width value)."
  [x :- Int8] :- Long
  (long (int (.val x))))

;; ================================================================
;; Promotion rules — Int8 promotes UP (widening) into the numeric tower
;; ================================================================

(defval TInt8)
(register-type-tag! Int8 (->TInt8))

;; convert TO Int8 (narrowing — truncates to a signed byte)
(deftm raster.types.promote/convert [t :- TInt8, x :- Int8] :- Int8 x)
(deftm raster.types.promote/convert [t :- TInt8, x :- Long] :- Int8
  (->Int8 (unchecked-byte x)))
(deftm raster.types.promote/convert [t :- TInt8, x :- Integer] :- Int8
  (->Int8 (unchecked-byte (long x))))

;; promote Int8 UP — the defining (widening) direction
(deftm raster.types.promote/convert [t :- TInteger, x :- Int8] :- Integer
  (int (.val x)))
(deftm raster.types.promote/convert [t :- TLong, x :- Int8] :- Long
  (long (.val x)))

;; lattice: Int8 mixed with a wider type widens to that type (never stays int8)
(register-promote-rule! Int8 Integer Integer)
(register-promote-rule! Int8 Long Long)
(register-promote-rule! Int8 Float Float)
(register-promote-rule! Int8 Double Double)

;; ================================================================
;; Array storage
;; ================================================================
;; Int8 is a SCALAR semantic type; int8 *arrays* are `(Array byte)` (byte[]).
;; We intentionally do NOT register a distinct `(Array Int8)` array type: byte[]
;; is already `(Array byte)`, and runtime array dispatch keys on the array's Java
;; class ([B) — it cannot distinguish `(Array Int8)` from `(Array byte)` (both are
;; [B), unlike Float16→short[] where short[] is otherwise unclaimed. This is also
;; the representation the quantized kernels want (the wasm/CPU int8 dot runs on
;; byte[] + explicit widening). Pull an int8 out with `(int8 (raster.arrays/aget
;; bs i))` or widen directly with `(int8->long (int8 (aget bs i)))`.

(deftm aget-widen
  "Read byte[] element i and widen (sign-extend) to a long — the int8 load used by
   the quantized dot. Equivalent to (int8->long (int8 (aget bs i))) but direct."
  [bs :- (Array byte), i :- Long] :- Long
  (long (raster.arrays/aget bs i)))
