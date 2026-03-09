(ns raster.math
  "Julia Base math functions as deftm generic functions.

  Provides typed dispatch for trig, exp/log, rounding, and other math
  functions. Both Double and Float variants are included.

  Usage:
    (require '[raster.math :refer [sin cos tan exp log sqrt]])
    (sin 1.0)         ;; => 0.8414709848078965
    (sin (float 1.0)) ;; => (float 0.84147096)"
  (:require [raster.core :refer [deftm]]))

;; ================================================================
;; Trigonometric functions
;; ================================================================

(deftm sin "Compute the sine of x (radians)."
  [x :- Double] :- Double (Math/sin x))
(deftm sin [x :- Float] :- Float (float (Math/sin (double x))))

(deftm cos "Compute the cosine of x (radians)."
  [x :- Double] :- Double (Math/cos x))
(deftm cos [x :- Float] :- Float (float (Math/cos (double x))))

(deftm tan "Compute the tangent of x (radians)."
  [x :- Double] :- Double (Math/tan x))
(deftm tan [x :- Float] :- Float (float (Math/tan (double x))))

(deftm asin "Compute the arc sine of x. Returns radians in [-pi/2, pi/2]."
  [x :- Double] :- Double (Math/asin x))
(deftm asin [x :- Float] :- Float (float (Math/asin (double x))))

(deftm acos "Compute the arc cosine of x. Returns radians in [0, pi]."
  [x :- Double] :- Double (Math/acos x))
(deftm acos [x :- Float] :- Float (float (Math/acos (double x))))

(deftm atan "Compute the arc tangent of x. Returns radians in [-pi/2, pi/2]."
  [x :- Double] :- Double (Math/atan x))
(deftm atan [x :- Float] :- Float (float (Math/atan (double x))))

(deftm atan2 "Compute the two-argument arc tangent of y/x. Returns radians in [-pi, pi]."
  [y :- Double, x :- Double] :- Double (Math/atan2 y x))
(deftm atan2 [y :- Float, x :- Float] :- Float (float (Math/atan2 (double y) (double x))))

;; ================================================================
;; Exponential and logarithmic functions
;; ================================================================

(deftm exp "Compute e^x."
  [x :- Double] :- Double (Math/exp x))
(deftm exp [x :- Float] :- Float (float (Math/exp (double x))))

(deftm log "Compute the natural logarithm of x."
  [x :- Double] :- Double (Math/log x))
(deftm log [x :- Float] :- Float (float (Math/log (double x))))

(deftm log2 "Compute the base-2 logarithm of x."
  [x :- Double] :- Double (/ (Math/log x) (Math/log 2.0)))
(deftm log2 [x :- Float] :- Float (float (/ (Math/log (double x)) (Math/log 2.0))))

(deftm log10 "Compute the base-10 logarithm of x."
  [x :- Double] :- Double (Math/log10 x))
(deftm log10 [x :- Float] :- Float (float (Math/log10 (double x))))

(deftm expm1 "Compute e^x - 1, accurate for small x."
  [x :- Double] :- Double (Math/expm1 x))
(deftm expm1 [x :- Float] :- Float (float (Math/expm1 (double x))))

(deftm log1p "Compute log(1 + x), accurate for small x."
  [x :- Double] :- Double (Math/log1p x))
(deftm log1p [x :- Float] :- Float (float (Math/log1p (double x))))

(deftm exp2
  "Compute 2^x."
  [x :- Double] :- Double
  (Math/pow 2.0 x))

(deftm exp2 [x :- Float] :- Float
  (float (Math/pow 2.0 (double x))))

(deftm exp10
  "Compute 10^x."
  [x :- Double] :- Double
  (Math/pow 10.0 x))

(deftm exp10 [x :- Float] :- Float
  (float (Math/pow 10.0 (double x))))

;; ================================================================
;; Rounding functions
;; ================================================================

(deftm ceil "Round x up to the nearest integer (toward positive infinity)."
  [x :- Double] :- Double (Math/ceil x))
(deftm ceil [x :- Float] :- Float (float (Math/ceil (double x))))

(deftm floor "Round x down to the nearest integer (toward negative infinity)."
  [x :- Double] :- Double (Math/floor x))
(deftm floor [x :- Float] :- Float (float (Math/floor (double x))))

(deftm round "Round x to the nearest integer (half-up)."
  [x :- Double] :- Long (Math/round x))
(deftm round [x :- Float] :- Integer (Math/round (float x)))

(deftm trunc
  "Truncate towards zero (round towards zero)."
  [x :- Double] :- Double
  (if (pos? x) (Math/floor x) (Math/ceil x)))

(deftm trunc [x :- Float] :- Float
  (float (if (pos? x) (Math/floor (double x)) (Math/ceil (double x)))))

;; ================================================================
;; Other math functions
;; ================================================================

(deftm hypot "Compute sqrt(x^2 + y^2) without intermediate overflow."
  [x :- Double, y :- Double] :- Double (Math/hypot x y))
(deftm hypot [x :- Float, y :- Float] :- Float (float (Math/hypot (double x) (double y))))

(deftm cbrt "Compute the cube root of x."
  [x :- Double] :- Double (Math/cbrt x))
(deftm cbrt [x :- Float] :- Float (float (Math/cbrt (double x))))

(deftm sinh "Compute the hyperbolic sine of x."
  [x :- Double] :- Double (Math/sinh x))
(deftm sinh [x :- Float] :- Float (float (Math/sinh (double x))))

(deftm cosh "Compute the hyperbolic cosine of x."
  [x :- Double] :- Double (Math/cosh x))
(deftm cosh [x :- Float] :- Float (float (Math/cosh (double x))))

(deftm tanh "Compute the hyperbolic tangent of x."
  [x :- Double] :- Double (Math/tanh x))
(deftm tanh [x :- Float] :- Float (float (Math/tanh (double x))))

;; ================================================================
;; Inverse hyperbolic functions
;; ================================================================

(deftm asinh "Compute the inverse hyperbolic sine of x."
  [x :- Double] :- Double
  (Math/log (+ x (Math/sqrt (+ (* x x) 1.0)))))
(deftm asinh [x :- Float] :- Float
  (float (Math/log (+ (double x) (Math/sqrt (+ (* (double x) (double x)) 1.0))))))

(deftm acosh "Compute the inverse hyperbolic cosine of x. Requires x >= 1."
  [x :- Double] :- Double
  (Math/log (+ x (Math/sqrt (- (* x x) 1.0)))))
(deftm acosh [x :- Float] :- Float
  (float (Math/log (+ (double x) (Math/sqrt (- (* (double x) (double x)) 1.0))))))

(deftm atanh "Compute the inverse hyperbolic tangent of x. Requires |x| < 1."
  [x :- Double] :- Double
  (* 0.5 (Math/log (/ (+ 1.0 x) (- 1.0 x)))))
(deftm atanh [x :- Float] :- Float
  (float (* 0.5 (Math/log (/ (+ 1.0 (double x)) (- 1.0 (double x)))))))

;; ================================================================
;; Degree/radian conversion
;; ================================================================

(deftm deg2rad "Convert degrees to radians."
  [x :- Double] :- Double (Math/toRadians x))
(deftm deg2rad [x :- Float] :- Float (float (Math/toRadians (double x))))

(deftm rad2deg "Convert radians to degrees."
  [x :- Double] :- Double (Math/toDegrees x))
(deftm rad2deg [x :- Float] :- Float (float (Math/toDegrees (double x))))

;; ================================================================
;; Special floating-point values
;; ================================================================

(deftm nextfloat "Return the next representable floating-point value above x."
  [x :- Double] :- Double (Math/nextUp x))
(deftm nextfloat [x :- Float] :- Float (Math/nextUp (float x)))

(deftm prevfloat "Return the next representable floating-point value below x."
  [x :- Double] :- Double (Math/nextDown x))
(deftm prevfloat [x :- Float] :- Float (Math/nextDown (float x)))

(deftm eps "Return the unit in the last place (ULP) of x -- the spacing to the next float."
  [x :- Double] :- Double (Math/ulp x))
(deftm eps [x :- Float] :- Float (Math/ulp (float x)))

;; ================================================================
;; Clamping
;; ================================================================

(deftm clamp "Clamp x to the interval [lo, hi]."
  [x :- Double, lo :- Double, hi :- Double] :- Double
  (Math/clamp x lo hi))
(deftm clamp [x :- Float, lo :- Float, hi :- Float] :- Float
  (float (Math/clamp (double x) (double lo) (double hi))))
(deftm clamp [x :- Long, lo :- Long, hi :- Long] :- Long
  (Math/clamp x lo hi))

;; ================================================================
;; Fused multiply-add (hardware FMA on modern CPUs)
;; ================================================================

(deftm fma "Compute x*y + z with a single rounding step (fused multiply-add)."
  [x :- Double, y :- Double, z :- Double] :- Double (Math/fma x y z))
(deftm fma [x :- Float, y :- Float, z :- Float] :- Float (Math/fma (float x) (float y) (float z)))

;; ================================================================
;; Copy sign
;; ================================================================

(deftm copysign "Return x with the sign of y."
  [x :- Double, y :- Double] :- Double (Math/copySign x y))
(deftm copysign [x :- Float, y :- Float] :- Float (Math/copySign (float x) (float y)))

;; ================================================================
;; Flip sign
;; ================================================================

(deftm flipsign
  "Return x with sign flipped if y is negative."
  [x :- Double, y :- Double] :- Double
  (if (neg? y) (- x) x))

(deftm flipsign [x :- Float, y :- Float] :- Float
  (float (if (neg? y) (- (double x)) (double x))))

;; ================================================================
;; Sincos (convenience — returns vector [sin cos])
;; ================================================================

(deftm sincos "Return [sin(x) cos(x)] as a vector."
  [x :- Double] [(Math/sin x) (Math/cos x)])

;; ================================================================
;; Signum (float-valued sign: -1.0, 0.0, or +1.0)
;; ================================================================

(deftm signum "Return the float-valued sign: -1.0, 0.0, or +1.0."
  [x :- Double] :- Double (Math/signum x))
(deftm signum [x :- Float] :- Float (Math/signum (float x)))
