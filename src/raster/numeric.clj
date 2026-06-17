(ns raster.numeric
  "Polymorphic arithmetic for raster typed dispatch.

  Provides +, -, *, / as deftm generic functions that dispatch
  on argument types. Extend with deftm for custom numeric types.

  Usage:
    (ns my.solver
      (:refer-clojure :exclude [+ - * / zero? min max abs])
      (:require [raster.numeric :refer [+ - * / zero]]))"
  (:refer-clojure :exclude [+ - * / zero? min max abs sqrt pow
                            mod rem quot
                            bit-and bit-or bit-xor bit-not
                            bit-shift-left bit-shift-right
                            unsigned-bit-shift-right
                            < <= > >= ==
                            even? odd? pos? neg?
                            infinite?
                            aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.compiler.core.dispatch :refer [set-reducible!]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.types.promote :as promote]))

;; Helper: call through dispatch without walker devirtualization.
;; The walker only devirtualizes direct (deftm-fn arg ...) calls.
;; Wrapping in a plain defn prevents the walker from resolving
;; the recursive (op px py) to a wrong specific method.
(defn- redispatch [op x y] (op x y))

;; ================================================================
;; Polymorphic addition
;; ================================================================

(deftm + "Add two numbers. Dispatches on argument types with promotion fallback."
  [x :- Long, y :- Long] :- Long (clojure.core/+ x y))
(deftm + [x :- Double, y :- Double] :- Double (clojure.core/+ x y))
(deftm + [x :- Long, y :- Double] :- Double (clojure.core/+ x y))
(deftm + [x :- Double, y :- Long] :- Double (clojure.core/+ x y))
(deftm + [x :- Float, y :- Float] :- Float (float (clojure.core/+ x y)))
(deftm + [x :- Integer, y :- Integer] :- Integer (unchecked-add-int x y))
(deftm + [x :- Number, y :- Number] :- Number
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (clojure.core/+ x y)
      (redispatch + px py))
    (clojure.core/+ x y)))

;; ================================================================
;; Polymorphic subtraction
;; ================================================================

(deftm - "Negate or subtract. Unary negation and binary subtraction with type dispatch."
  [x :- Long] :- Long (clojure.core/- x))
(deftm - [x :- Double] :- Double (clojure.core/- x))
(deftm - [x :- Float] :- Float (float (clojure.core/- x)))
(deftm - [x :- Long, y :- Long] :- Long (clojure.core/- x y))
(deftm - [x :- Double, y :- Double] :- Double (clojure.core/- x y))
(deftm - [x :- Long, y :- Double] :- Double (clojure.core/- x y))
(deftm - [x :- Double, y :- Long] :- Double (clojure.core/- x y))
(deftm - [x :- Float, y :- Float] :- Float (float (clojure.core/- x y)))
(deftm - [x :- Integer] :- Integer (unchecked-negate-int x))
(deftm - [x :- Integer, y :- Integer] :- Integer (unchecked-subtract-int x y))
(deftm - [x :- Number, y :- Number] :- Number
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (clojure.core/- x y)
      (redispatch - px py))
    (clojure.core/- x y)))

;; ================================================================
;; Polymorphic multiplication
;; ================================================================

(deftm * "Multiply two numbers. Dispatches on argument types with promotion fallback."
  [x :- Long, y :- Long] :- Long (clojure.core/* x y))
(deftm * [x :- Double, y :- Double] :- Double (clojure.core/* x y))
(deftm * [x :- Long, y :- Double] :- Double (clojure.core/* x y))
(deftm * [x :- Double, y :- Long] :- Double (clojure.core/* x y))
(deftm * [x :- Float, y :- Float] :- Float (float (clojure.core/* x y)))
(deftm * [x :- Integer, y :- Integer] :- Integer (unchecked-multiply-int x y))
(deftm * [x :- Number, y :- Number] :- Number
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (clojure.core/* x y)
      (redispatch * px py))
    (clojure.core/* x y)))

;; ================================================================
;; Polymorphic division
;; ================================================================

(deftm / "Divide two numbers. Dispatches on argument types with promotion fallback."
  [x :- Long, y :- Long] :- Long (clojure.core// x y))
(deftm / [x :- Double, y :- Double] :- Double (clojure.core// x y))
(deftm / [x :- Long, y :- Double] :- Double (clojure.core// x y))
(deftm / [x :- Double, y :- Long] :- Double (clojure.core// x y))
(deftm / [x :- Float, y :- Float] :- Float (float (clojure.core// x y)))
(deftm / [x :- Integer, y :- Integer] :- Integer (unchecked-divide-int x y))
(deftm / [x :- Number, y :- Number] :- Number
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (clojure.core// x y)
      (redispatch / px py))
    (clojure.core// x y)))

;; ================================================================
;; Promotion-only fallbacks for non-Number types (defvalue types)
;; These are less specific than Number+Number, catching custom types
;; that don't extend java.lang.Number but have promote rules.
;; ================================================================

(deftm + [x :- Object, y :- Object]
  (cond
    ;; Fast path for array types (common in AD backward accumulation)
    (and (.isArray (class x)) (.isArray (class y))
         (identical? (class x) (class y)))
    (redispatch + x y)
    ;; Promotion path for custom types
    :else
    (if-let [[px py] (promote/promote x y)]
      (if (and (identical? (class px) (class x))
               (identical? (class py) (class y)))
        (throw (ex-info (str "No method for + with same-type " (class x))
                        {:types [(class x) (class y)]}))
        (redispatch + px py))
      (throw (ex-info (str "No promotion rule for + with types "
                           (.getSimpleName (class x)) " and "
                           (.getSimpleName (class y)))
                      {:types [(class x) (class y)]})))))

(deftm - [x :- Object, y :- Object]
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (throw (ex-info (str "No method for - with same-type " (class x))
                      {:types [(class x) (class y)]}))
      (redispatch - px py))
    (throw (ex-info (str "No promotion rule for - with types "
                         (.getSimpleName (class x)) " and "
                         (.getSimpleName (class y)))
                    {:types [(class x) (class y)]}))))

(deftm * [x :- Object, y :- Object]
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (throw (ex-info (str "No method for * with same-type " (class x))
                      {:types [(class x) (class y)]}))
      (redispatch * px py))
    (throw (ex-info (str "No promotion rule for * with types "
                         (.getSimpleName (class x)) " and "
                         (.getSimpleName (class y)))
                    {:types [(class x) (class y)]}))))

(deftm / [x :- Object, y :- Object]
  (if-let [[px py] (promote/promote x y)]
    (if (and (identical? (class px) (class x))
             (identical? (class py) (class y)))
      (throw (ex-info (str "No method for / with same-type " (class x))
                      {:types [(class x) (class y)]}))
      (redispatch / px py))
    (throw (ex-info (str "No promotion rule for / with types "
                         (.getSimpleName (class x)) " and "
                         (.getSimpleName (class y)))
                    {:types [(class x) (class y)]}))))

;; Mark +, *, - as reducible (variadic via fold)
(set-reducible! #'+)
(set-reducible! #'*)

;; ================================================================
;; Number tower: abs, sign, clamp, min, max, zero?
;; ================================================================

(deftm abs "Return the absolute value."
  [x :- Long] :- Long (Math/abs x))
(deftm abs [x :- Double] :- Double (Math/abs x))
(deftm abs [x :- Float] :- Float (Math/abs (float x)))
(deftm abs [x :- Integer] :- Integer (Math/abs (int x)))

(deftm sign "Return the sign of x: -1, 0, or 1."
  [x :- Long] :- Long (Long/signum x))
(deftm sign [x :- Double] :- Double (Math/signum x))

(deftm clamp "Clamp x to the interval [lo, hi]."
  [x :- Long, lo :- Long, hi :- Long] :- Long
  (Math/clamp x lo hi))
(deftm clamp [x :- Double, lo :- Double, hi :- Double] :- Double
  (Math/clamp x lo hi))

(deftm min "Return the smaller of two values."
  [x :- Long, y :- Long] :- Long (Math/min x y))
(deftm min [x :- Double, y :- Double] :- Double (Math/min x y))
(deftm min [x :- Float, y :- Float] :- Float (Math/min (float x) (float y)))
(deftm min [x :- Integer, y :- Integer] :- Integer (Math/min (int x) (int y)))

(deftm max "Return the larger of two values."
  [x :- Long, y :- Long] :- Long (Math/max x y))
(deftm max [x :- Double, y :- Double] :- Double (Math/max x y))
(deftm max [x :- Float, y :- Float] :- Float (Math/max (float x) (float y)))
(deftm max [x :- Integer, y :- Integer] :- Integer (Math/max (int x) (int y)))

(deftm sqrt "Return the square root of x."
  [x :- Double] :- Double (Math/sqrt x))
(deftm sqrt [x :- Float] :- Float (float (Math/sqrt (double x))))

(deftm pow "Raise x to the power n."
  [x :- Double, n :- Double] :- Double (Math/pow x n))
(deftm pow [x :- Double, n :- Long] :- Double (Math/pow x (double n)))
(deftm pow [x :- Float, n :- Float] :- Float (float (Math/pow (double x) (double n))))

(deftm fast-pow
  "Fast approximate x^n for x > 0, via 2^(n·log2 x) with degree-5 minimax-fit
  polynomials over the IEEE mantissa/exponent split. Max relative error
  ~2.4e-5 over x∈[1e-3,100], n∈[0.5,1.1].

  For perf-sensitive code where ~1e-5 accuracy is acceptable (e.g. SGD layout
  gradients, where the result is clipped anyway). Requires x > 0 — NOT a
  full-precision drop-in for pow. Avoids Math/pow's slow libm range reduction:
  ~15 FP ops + 2 bit casts, all inline-able."
  [x :- Double, n :- Double] :- Double
  (let [bits (Double/doubleToRawLongBits x)
        ;; log2(x) = unbiased-exponent + log2(mantissa), mantissa in [1,2)
        xexp (double (- (bit-and (unsigned-bit-shift-right bits 52) 0x7ff) 1023))
        mant (Double/longBitsToDouble
              (bit-or (bit-and bits 0xfffffffffffff) 0x3ff0000000000000))
        ;; Horner: log2(mant) ≈ ((((c5 m + c4) m + c3) m + c2) m + c1) m + c0
        lm   (+ (* (+ (* (+ (* (+ (* (+ (* 0.04342890785229833 mant)
                                        -0.4048671746887665) mant)
                                  1.5939013644311786) mant)
                            -3.4924942814448627) mant)
                      5.046876046257392) mant)
                -2.7868129542772677)
        pw   (* n (+ xexp lm))
        ;; 2^pw = 2^floor(pw) · 2^frac(pw)
        fi   (Math/floor pw)
        fr   (- pw fi)
        q    (+ (* (+ (* (+ (* (+ (* (+ (* 0.001894379423521989 fr)
                                        0.008940582529443615) fr)
                                  0.05587655686843709) fr)
                            0.24013169187238942) fr)
                      0.6931567766987405) fr)
                0.9999997696337165)
        scale (Double/longBitsToDouble
               (bit-shift-left (+ (long fi) 1023) 52))]
    (* q scale)))

;; Extract the real-valued part of any numeric type.
;; For plain numbers, identity. For Dual types, extracts .v.
;; Used by ODE step-size control (error norms should NOT be differentiated).
(deftm real-value "Extract the real-valued part. Identity for plain numbers; extracts .v for Dual types."
  [x :- Double] :- Double x)
(deftm real-value [x :- Long] :- Double (double x))
(deftm real-value [x :- Float] :- Double (double x))
(deftm real-value [x :- Number] :- Double (.doubleValue ^Number x))

;; Extract first derivative with respect to a seeded scalar direction.
;; For plain numbers this is always zero (constant w.r.t. the seed).
;; AD backends (Dual, Jet, ...) extend this operation with deftm methods.
(deftm derivative-value "Extract the derivative component. Returns 0.0 for plain numbers; AD types override."
  [x :- Double] :- Double 0.0)
(deftm derivative-value [x :- Long] :- Double 0.0)
(deftm derivative-value [x :- Float] :- Double 0.0)
(deftm derivative-value [x :- Number] :- Double 0.0)

(deftm zero? "Return true if x is zero."
  [x :- Long] (clojure.core/== x 0))
(deftm zero? [x :- Double] (clojure.core/== x 0.0))
(deftm zero? [x :- Float] (clojure.core/== x (float 0.0)))
(deftm zero? [x :- Integer] (clojure.core/== x 0))

;; ================================================================
;; Generic constructors
;; ================================================================

(deftm zero "Return the zero value for x's type. Also works on arrays (returns zero-filled copy)."
  [x :- Long] :- Long 0)
(deftm zero [x :- Double] :- Double 0.0)
(deftm zero [x :- Float] :- Float (float 0.0))
(deftm zero [x :- Integer] :- Integer (int 0))
(deftm zero [x :- (Array double)] :- (Array double) (double-array (alength x)))
(deftm zero [x :- (Array long)] :- (Array long) (long-array (alength x)))
(deftm zero [x :- (Array float)] :- (Array float) (float-array (alength x)))

(deftm one "Return the multiplicative identity for x's type."
  [x :- Long] :- Long 1)
(deftm one [x :- Double] :- Double 1.0)
(deftm one [x :- Float] :- Float (float 1.0))
(deftm one [x :- Integer] :- Integer (int 1))

;; Julia-style oftype: convert val to match ref's type.
;; Use in (All [T] ...) bodies to get type-correct constants:
;;   (oftype x 1.0) → (float 1.0) when x is Float or (Array float)
(deftm oftype "Convert val to match ref's type. Julia-style oftype for parametric code."
  [ref :- Double, val :- Double] :- Double val)
(deftm oftype [ref :- Float, val :- Double] :- Float (float val))
(deftm oftype [ref :- Float, val :- Float] :- Float val)
(deftm oftype [ref :- Double, val :- Float] :- Double (double val))
(deftm oftype [ref :- Long, val :- Long] :- Long val)
(deftm oftype [ref :- Integer, val :- Long] :- Integer (int val))
(deftm oftype [ref :- Float, val :- Long] :- Float (float val))
(deftm oftype [ref :- Double, val :- Long] :- Double (double val))
;; Array variants: extract element type from array reference
(deftm oftype [ref :- (Array double), val :- Double] :- Double val)
(deftm oftype [ref :- (Array float), val :- Double] :- Float (float val))
(deftm oftype [ref :- (Array float), val :- Float] :- Float val)
(deftm oftype [ref :- (Array double), val :- Float] :- Double (double val))
(deftm oftype [ref :- (Array long), val :- Long] :- Long val)
(deftm oftype [ref :- (Array int), val :- Long] :- Integer (int val))
(deftm oftype [ref :- (Array double), val :- Long] :- Double (double val))
(deftm oftype [ref :- (Array float), val :- Long] :- Float (float val))

(deftm similar "Allocate a new uninitialized array of the same type and length as a."
  [a :- (Array double)] :- (Array double) (double-array (alength a)))
(deftm similar [a :- (Array long)] :- (Array long) (long-array (alength a)))
(deftm similar [a :- (Array float)] :- (Array float) (float-array (alength a)))
(deftm similar [a :- (Array int)] :- (Array int) (int-array (alength a)))
(deftm similar [a :- objects] :- objects (make-array Object (alength a)))

(deftm zero [x :- (Array int)] :- (Array int) (int-array (alength x)))

;; Type-dispatched infinity values (for generic amax/amin accumulators)
(deftm neg-inf-val "Return negative infinity for x's type. Used as initial accumulator for max reductions."
  [x :- Double] :- Double Double/NEGATIVE_INFINITY)
(deftm neg-inf-val [x :- Float] :- Float Float/NEGATIVE_INFINITY)
(deftm pos-inf-val "Return positive infinity for x's type. Used as initial accumulator for min reductions."
  [x :- Double] :- Double Double/POSITIVE_INFINITY)
(deftm pos-inf-val [x :- Float] :- Float Float/POSITIVE_INFINITY)

;; ================================================================
;; Array element-wise arithmetic — parametric over element type.
;; Allocating ops overload +, -, *, / on array types.
;; ================================================================

;; --- array + array ---
(deftm + [a :- (Array double), b :- (Array double)] :- (Array double)
  (let [n (Math/min (long (alength a)) (long (alength b)))
        out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/+ (aget a i) (aget b i))))
    out))
(deftm + [a :- (Array float), b :- (Array float)] :- (Array float)
  (let [n (Math/min (long (alength a)) (long (alength b)))
        out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/+ (aget a i) (aget b i))))
    out))

;; --- scalar * array ---
(deftm * [c :- Double, a :- (Array double)] :- (Array double)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/* c (aget a i)))) out))
(deftm * [a :- (Array double), c :- Double] :- (Array double)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/* (aget a i) c))) out))
(deftm * [c :- Float, a :- (Array float)] :- (Array float)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/* c (aget a i)))) out))
(deftm * [a :- (Array float), c :- Float] :- (Array float)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/* (aget a i) c))) out))

;; --- array negation / subtraction ---
(deftm - [a :- (Array double)] :- (Array double)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/- (aget a i)))) out))
(deftm - [a :- (Array float)] :- (Array float)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/- (aget a i)))) out))
(deftm - [a :- (Array double), b :- (Array double)] :- (Array double)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/- (aget a i) (aget b i)))) out))
(deftm - [a :- (Array float), b :- (Array float)] :- (Array float)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core/- (aget a i) (aget b i)))) out))

;; --- array / scalar ---
(deftm / [a :- (Array double), c :- Double] :- (Array double)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core// (aget a i) c))) out))
(deftm / [a :- (Array float), c :- Float] :- (Array float)
  (let [n (alength a) out (similar a)]
    (dotimes [i n] (aset out i (clojure.core// (aget a i) c))) out))

;; ================================================================
;; In-place array operations — parametric
;; ================================================================

(deftm add! "Element-wise a + b, storing result in out. Returns out."
  (All [T] [out :- (Array T), a :- (Array T), b :- (Array T)]
       :- (Array T)
       (let [n (alength out)]
         (dotimes [i n] (aset out i (clojure.core/+ (aget a i) (aget b i))))
         out)))

(deftm sub! "Element-wise a - b, storing result in out. Returns out."
  (All [T] [out :- (Array T), a :- (Array T), b :- (Array T)]
       :- (Array T)
       (let [n (alength out)]
         (dotimes [i n] (aset out i (clojure.core/- (aget a i) (aget b i))))
         out)))

(deftm scale! "Scale array a by scalar c, storing result in out. Returns out."
  (All [T] [out :- (Array T), c :- Double, a :- (Array T)]
       :- (Array T)
       (let [n (alength out)]
         (dotimes [i n] (aset out i (clojure.core/* c (aget a i))))
         out)))

(deftm axpy! "Compute a + c*b element-wise, storing result in out. Returns out."
  (All [T] [out :- (Array T), a :- (Array T), c :- Double, b :- (Array T)]
       :- (Array T)
       (let [n (alength out)]
         (dotimes [i n] (aset out i (clojure.core/+ (aget a i) (clojure.core/* c (aget b i)))))
         out)))

;; ================================================================
;; Integer division and modulo
;; ================================================================

(deftm mod "Return x mod y (floored modulus, always non-negative for positive y)."
  [x :- Long, y :- Long] :- Long (clojure.core/mod x y))
(deftm mod [x :- Integer, y :- Integer] :- Integer (int (clojure.core/mod x y)))
(deftm mod [x :- Double, y :- Double] :- Double (clojure.core/mod x y))
(deftm mod [x :- Float, y :- Float] :- Float (float (clojure.core/mod x y)))

(deftm rem "Return the remainder of x / y (truncated division)."
  [x :- Long, y :- Long] :- Long (clojure.core/rem x y))
(deftm rem [x :- Integer, y :- Integer] :- Integer (int (clojure.core/rem x y)))
(deftm rem [x :- Double, y :- Double] :- Double (clojure.core/rem x y))

(deftm quot "Return the integer quotient of x / y (truncated toward zero)."
  [x :- Long, y :- Long] :- Long (clojure.core/quot x y))
(deftm quot [x :- Integer, y :- Integer] :- Integer (unchecked-divide-int x y))
(deftm quot [x :- Double, y :- Double] :- Double (double (clojure.core/quot x y)))

;; ================================================================
;; Bitwise operations
;; ================================================================

(deftm bit-and "Bitwise AND of x and y."
  [x :- Long, y :- Long] :- Long (clojure.core/bit-and x y))
(deftm bit-or "Bitwise OR of x and y."
  [x :- Long, y :- Long] :- Long (clojure.core/bit-or x y))
(deftm bit-xor "Bitwise XOR of x and y."
  [x :- Long, y :- Long] :- Long (clojure.core/bit-xor x y))
(deftm bit-not "Bitwise complement of x."
  [x :- Long] :- Long (clojure.core/bit-not x))
(deftm bit-shift-left "Shift x left by n bits."
  [x :- Long, n :- Long] :- Long (clojure.core/bit-shift-left x n))
(deftm bit-shift-right "Arithmetic shift x right by n bits (sign-extending)."
  [x :- Long, n :- Long] :- Long (clojure.core/bit-shift-right x n))
(deftm unsigned-bit-shift-right "Logical shift x right by n bits (zero-filling)."
  [x :- Long, n :- Long] :- Long
  (clojure.core/unsigned-bit-shift-right x n))

;; Integer variants
(deftm bit-and [x :- Integer, y :- Integer] :- Integer
  (int (clojure.core/bit-and (int x) (int y))))
(deftm bit-or  [x :- Integer, y :- Integer] :- Integer
  (int (clojure.core/bit-or (int x) (int y))))
(deftm bit-xor [x :- Integer, y :- Integer] :- Integer
  (int (clojure.core/bit-xor (int x) (int y))))
(deftm bit-not [x :- Integer] :- Integer
  (int (clojure.core/bit-not (int x))))

;; ================================================================
;; Comparison operators
;; ================================================================

(deftm < "Return true if x is strictly less than y."
  [x :- Long, y :- Long] :- Boolean (clojure.core/< x y))
(deftm < [x :- Double, y :- Double] :- Boolean (clojure.core/< x y))
(deftm < [x :- Float, y :- Float] :- Boolean (clojure.core/< x y))
(deftm < [x :- Integer, y :- Integer] :- Boolean (clojure.core/< x y))

(deftm <= "Return true if x is less than or equal to y."
  [x :- Long, y :- Long] :- Boolean (clojure.core/<= x y))
(deftm <= [x :- Double, y :- Double] :- Boolean (clojure.core/<= x y))
(deftm <= [x :- Float, y :- Float] :- Boolean (clojure.core/<= x y))
(deftm <= [x :- Integer, y :- Integer] :- Boolean (clojure.core/<= x y))

(deftm > "Return true if x is strictly greater than y."
  [x :- Long, y :- Long] :- Boolean (clojure.core/> x y))
(deftm > [x :- Double, y :- Double] :- Boolean (clojure.core/> x y))
(deftm > [x :- Float, y :- Float] :- Boolean (clojure.core/> x y))
(deftm > [x :- Integer, y :- Integer] :- Boolean (clojure.core/> x y))

(deftm >= "Return true if x is greater than or equal to y."
  [x :- Long, y :- Long] :- Boolean (clojure.core/>= x y))
(deftm >= [x :- Double, y :- Double] :- Boolean (clojure.core/>= x y))
(deftm >= [x :- Float, y :- Float] :- Boolean (clojure.core/>= x y))
(deftm >= [x :- Integer, y :- Integer] :- Boolean (clojure.core/>= x y))

(deftm == "Return true if x and y are numerically equal."
  [x :- Long, y :- Long] :- Boolean (clojure.core/== x y))
(deftm == [x :- Double, y :- Double] :- Boolean (clojure.core/== x y))
(deftm == [x :- Float, y :- Float] :- Boolean (clojure.core/== x y))
(deftm == [x :- Integer, y :- Integer] :- Boolean (clojure.core/== x y))

;; Cross-type comparisons via promotion
(deftm < [x :- Number, y :- Number] :- Boolean
  (if-let [[px py] (promote/promote x y)]
    (clojure.core/< (.doubleValue ^Number px) (.doubleValue ^Number py))
    (clojure.core/< (.doubleValue ^Number x) (.doubleValue ^Number y))))
(deftm <= [x :- Number, y :- Number] :- Boolean
  (if-let [[px py] (promote/promote x y)]
    (clojure.core/<= (.doubleValue ^Number px) (.doubleValue ^Number py))
    (clojure.core/<= (.doubleValue ^Number x) (.doubleValue ^Number y))))
(deftm > [x :- Number, y :- Number] :- Boolean
  (if-let [[px py] (promote/promote x y)]
    (clojure.core/> (.doubleValue ^Number px) (.doubleValue ^Number py))
    (clojure.core/> (.doubleValue ^Number x) (.doubleValue ^Number y))))
(deftm >= [x :- Number, y :- Number] :- Boolean
  (if-let [[px py] (promote/promote x y)]
    (clojure.core/>= (.doubleValue ^Number px) (.doubleValue ^Number py))
    (clojure.core/>= (.doubleValue ^Number x) (.doubleValue ^Number y))))

;; ================================================================
;; Numeric predicates
;; ================================================================

(deftm even? "Return true if x is even."
  [x :- Long] :- Boolean (clojure.core/even? x))
(deftm even? [x :- Integer] :- Boolean (clojure.core/even? (int x)))
(deftm odd? "Return true if x is odd."
  [x :- Long] :- Boolean (clojure.core/odd? x))
(deftm odd?  [x :- Integer] :- Boolean (clojure.core/odd? (int x)))
(deftm pos? "Return true if x is positive."
  [x :- Long] :- Boolean (clojure.core/pos? x))
(deftm pos?  [x :- Double] :- Boolean (clojure.core/pos? x))
(deftm pos?  [x :- Float] :- Boolean (clojure.core/pos? (float x)))
(deftm pos?  [x :- Integer] :- Boolean (clojure.core/pos? (int x)))
(deftm neg? "Return true if x is negative."
  [x :- Long] :- Boolean (clojure.core/neg? x))
(deftm neg?  [x :- Double] :- Boolean (clojure.core/neg? x))
(deftm neg?  [x :- Float] :- Boolean (clojure.core/neg? (float x)))
(deftm neg?  [x :- Integer] :- Boolean (clojure.core/neg? (int x)))

;; ================================================================
;; Integer exponentiation
;; ================================================================

(deftm pow [x :- Long, n :- Long] :- Long
  (loop [base (long x) exp (long n) acc (long 1)]
    (cond
      (clojure.core/== exp 0) acc
      (clojure.core/odd? exp) (recur (clojure.core/* base base)
                                     (clojure.core/quot (clojure.core/dec exp) 2)
                                     (clojure.core/* acc base))
      :else (recur (clojure.core/* base base)
                   (clojure.core/quot exp 2)
                   acc))))

;; ================================================================
;; Floating-point constants
;; ================================================================

(def ^{:const true :tag 'double} neg-inf Double/NEGATIVE_INFINITY)
(def ^{:const true :tag 'double} pos-inf Double/POSITIVE_INFINITY)
(def ^{:const true :tag 'double} nan-val Double/NaN)
(def ^{:const true :tag 'double} pi Math/PI)
(def ^{:const true :tag 'double} e Math/E)

;; ================================================================
;; Floating-point predicates
;; ================================================================

(deftm nan? "Return true if x is NaN."
  [x :- Double] :- Boolean (Double/isNaN x))
(deftm nan? [x :- Float] :- Boolean (Float/isNaN x))

(deftm infinite? "Return true if x is positive or negative infinity."
  [x :- Double] :- Boolean (Double/isInfinite x))
(deftm infinite? [x :- Float] :- Boolean (Float/isInfinite x))

(deftm finite? "Return true if x is neither NaN nor infinite."
  [x :- Double] :- Boolean (Double/isFinite x))
(deftm finite? [x :- Float] :- Boolean (Float/isFinite x))

;; ================================================================
;; Type cast wrappers (GPU-compilable)
;; ================================================================

(deftm to-double "Convert x to Double."
  [x :- Double] :- Double x)
(deftm to-double [x :- Long] :- Double (double x))
(deftm to-double [x :- Float] :- Double (double x))
(deftm to-double [x :- Integer] :- Double (double x))

(deftm to-long "Convert x to Long (truncates floating-point values)."
  [x :- Long] :- Long x)
(deftm to-long [x :- Double] :- Long (long x))
(deftm to-long [x :- Integer] :- Long (long x))

(deftm to-float "Convert x to Float (may lose precision from Double)."
  [x :- Float] :- Float x)
(deftm to-float [x :- Double] :- Float (float x))
(deftm to-float [x :- Long] :- Float (float x))

(deftm to-int "Convert x to Integer (truncates from Long/Double)."
  [x :- Integer] :- Integer x)
(deftm to-int [x :- Long] :- Integer (int x))
(deftm to-int [x :- Double] :- Integer (int x))

;; ================================================================
;; Vector arithmetic (PersistentVector)
;; Enables generic ODE solvers with vector state, e.g. Dual number
;; sensitivity analysis through standard solve(RK4, ...).
;; ================================================================

(deftm + [a :- clojure.lang.PersistentVector, b :- clojure.lang.PersistentVector]
  (mapv + a b))
(deftm - [a :- clojure.lang.PersistentVector]
  (mapv - a))
(deftm - [a :- clojure.lang.PersistentVector, b :- clojure.lang.PersistentVector]
  (mapv - a b))
(deftm * [a :- Double, b :- clojure.lang.PersistentVector]
  (mapv #(* a %) b))
(deftm * [a :- clojure.lang.PersistentVector, b :- Double]
  (mapv #(* % b) a))
