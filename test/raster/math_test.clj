(ns raster.math-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.math :refer [sin cos tan asin acos atan atan2
                                 exp log log2 log10 expm1 log1p exp2 exp10
                                 ceil floor round trunc
                                 hypot cbrt
                                 sinh cosh tanh asinh acosh atanh
                                 deg2rad rad2deg
                                 nextfloat prevfloat eps
                                 clamp fma copysign flipsign
                                 sincos signum]]))

(def ^:private tol 1e-12)
(def ^:private float-tol 1e-5)

(defn- approx=
  "Check approximate equality within tolerance."
  ([expected actual] (approx= expected actual tol))
  ([expected actual tolerance]
   (< (Math/abs (double (- (double expected) (double actual)))) tolerance)))

;; ================================================================
;; Trigonometric functions
;; ================================================================

(deftest sin-test
  (testing "sin at known values"
    (is (approx= 0.0 (sin 0.0)))
    (is (approx= 1.0 (sin (/ Math/PI 2.0))))
    (is (approx= 0.0 (sin Math/PI)))
    (is (approx= -1.0 (sin (* 1.5 Math/PI)))))
  (testing "sin Float variant"
    (is (approx= 0.0 (sin (float 0.0)) float-tol))
    (is (approx= 1.0 (sin (float (/ Math/PI 2.0))) float-tol))
    (is (instance? Float (sin (float 0.0))))))

(deftest cos-test
  (testing "cos at known values"
    (is (approx= 1.0 (cos 0.0)))
    (is (approx= 0.0 (cos (/ Math/PI 2.0))))
    (is (approx= -1.0 (cos Math/PI)))
    (is (approx= 0.0 (cos (* 1.5 Math/PI)))))
  (testing "cos Float variant"
    (is (approx= 1.0 (cos (float 0.0)) float-tol))
    (is (instance? Float (cos (float 0.0))))))

(deftest tan-test
  (testing "tan at known values"
    (is (approx= 0.0 (tan 0.0)))
    (is (approx= 1.0 (tan (/ Math/PI 4.0)))))
  (testing "tan Float variant"
    (is (approx= 0.0 (tan (float 0.0)) float-tol))
    (is (instance? Float (tan (float 0.0))))))

(deftest asin-test
  (testing "asin at known values"
    (is (approx= 0.0 (asin 0.0)))
    (is (approx= (/ Math/PI 2.0) (asin 1.0)))
    (is (approx= (/ Math/PI -2.0) (asin -1.0))))
  (testing "asin Float variant"
    (is (approx= 0.0 (asin (float 0.0)) float-tol))
    (is (instance? Float (asin (float 0.0))))))

(deftest acos-test
  (testing "acos at known values"
    (is (approx= (/ Math/PI 2.0) (acos 0.0)))
    (is (approx= 0.0 (acos 1.0)))
    (is (approx= Math/PI (acos -1.0))))
  (testing "acos Float variant"
    (is (instance? Float (acos (float 0.0))))))

(deftest atan-test
  (testing "atan at known values"
    (is (approx= 0.0 (atan 0.0)))
    (is (approx= (/ Math/PI 4.0) (atan 1.0))))
  (testing "atan Float variant"
    (is (instance? Float (atan (float 0.0))))))

(deftest atan2-test
  (testing "atan2 at known values"
    (is (approx= 0.0 (atan2 0.0 1.0)))
    (is (approx= (/ Math/PI 2.0) (atan2 1.0 0.0)))
    (is (approx= Math/PI (atan2 0.0 -1.0))))
  (testing "atan2 Float variant"
    (is (instance? Float (atan2 (float 0.0) (float 1.0))))))

;; ================================================================
;; Inverse trig roundtrips
;; ================================================================

(deftest trig-roundtrip-test
  (testing "sin/asin roundtrip"
    (doseq [x [0.0 0.5 -0.5 0.9 -0.9]]
      (is (approx= x (sin (asin x))))))
  (testing "cos/acos roundtrip"
    (doseq [x [0.0 0.5 1.0]]
      (is (approx= x (cos (acos x))))))
  (testing "tan/atan roundtrip"
    (doseq [x [0.0 1.0 -1.0 0.5]]
      (is (approx= x (tan (atan x)))))))

;; ================================================================
;; Exponential and logarithmic functions
;; ================================================================

(deftest exp-test
  (testing "exp at known values"
    (is (approx= 1.0 (exp 0.0)))
    (is (approx= Math/E (exp 1.0))))
  (testing "exp Float variant"
    (is (approx= 1.0 (exp (float 0.0)) float-tol))
    (is (instance? Float (exp (float 0.0))))))

(deftest log-test
  (testing "log at known values"
    (is (approx= 0.0 (log 1.0)))
    (is (approx= 1.0 (log Math/E))))
  (testing "exp/log roundtrip"
    (doseq [x [0.5 1.0 2.0 10.0]]
      (is (approx= x (log (exp x))))))
  (testing "log Float variant"
    (is (approx= 0.0 (log (float 1.0)) float-tol))
    (is (instance? Float (log (float 1.0))))))

(deftest log-edge-cases-test
  (testing "log(0) is -Infinity"
    (is (Double/isInfinite (log 0.0)))
    (is (neg? (log 0.0))))
  (testing "exp(large) is Infinity"
    (is (Double/isInfinite (exp 1000.0)))))

(deftest log2-test
  (testing "log2 at powers of 2"
    (is (approx= 0.0 (log2 1.0)))
    (is (approx= 1.0 (log2 2.0)))
    (is (approx= 2.0 (log2 4.0)))
    (is (approx= 3.0 (log2 8.0)))
    (is (approx= 10.0 (log2 1024.0))))
  (testing "log2 Float variant"
    (is (approx= 1.0 (log2 (float 2.0)) float-tol))
    (is (instance? Float (log2 (float 2.0))))))

(deftest log10-test
  (testing "log10 at powers of 10"
    (is (approx= 0.0 (log10 1.0)))
    (is (approx= 1.0 (log10 10.0)))
    (is (approx= 2.0 (log10 100.0)))
    (is (approx= 3.0 (log10 1000.0))))
  (testing "log10 Float variant"
    (is (approx= 1.0 (log10 (float 10.0)) float-tol))
    (is (instance? Float (log10 (float 10.0))))))

(deftest expm1-test
  (testing "expm1 near zero is more accurate than (- (exp x) 1)"
    (is (approx= 0.0 (expm1 0.0)))
    ;; expm1(1e-15) should be very close to 1e-15
    (is (approx= 1e-15 (expm1 1e-15) 1e-25)))
  (testing "expm1 Float variant"
    (is (instance? Float (expm1 (float 0.0))))))

(deftest log1p-test
  (testing "log1p near zero"
    (is (approx= 0.0 (log1p 0.0)))
    ;; log1p(1e-15) should be very close to 1e-15
    (is (approx= 1e-15 (log1p 1e-15) 1e-25)))
  (testing "log1p Float variant"
    (is (instance? Float (log1p (float 0.0))))))

(deftest exp2-test
  (testing "exp2 at known values"
    (is (approx= 1.0 (exp2 0.0)))
    (is (approx= 2.0 (exp2 1.0)))
    (is (approx= 4.0 (exp2 2.0)))
    (is (approx= 8.0 (exp2 3.0)))
    (is (approx= 0.5 (exp2 -1.0))))
  (testing "exp2 Float variant"
    (is (approx= 4.0 (exp2 (float 2.0)) float-tol))
    (is (instance? Float (exp2 (float 2.0))))))

(deftest exp10-test
  (testing "exp10 at known values"
    (is (approx= 1.0 (exp10 0.0)))
    (is (approx= 10.0 (exp10 1.0)))
    (is (approx= 100.0 (exp10 2.0)))
    (is (approx= 0.1 (exp10 -1.0))))
  (testing "exp10 Float variant"
    (is (approx= 100.0 (exp10 (float 2.0)) float-tol))
    (is (instance? Float (exp10 (float 2.0))))))

;; ================================================================
;; Rounding functions
;; ================================================================

(deftest ceil-test
  (testing "ceil at various values"
    (is (= 3.0 (ceil 2.3)))
    (is (= -2.0 (ceil -2.3)))
    (is (= 1.0 (ceil 0.1)))
    (is (= 0.0 (ceil 0.0)))
    (is (= 5.0 (ceil 5.0))))
  (testing "ceil Float variant"
    (is (= (float 3.0) (ceil (float 2.3))))
    (is (instance? Float (ceil (float 2.3))))))

(deftest floor-test
  (testing "floor at various values"
    (is (= 2.0 (floor 2.9)))
    (is (= -3.0 (floor -2.3)))
    (is (= 0.0 (floor 0.9)))
    (is (= 0.0 (floor 0.0)))
    (is (= 5.0 (floor 5.0))))
  (testing "floor Float variant"
    (is (= (float 2.0) (floor (float 2.9))))
    (is (instance? Float (floor (float 2.9))))))

(deftest round-test
  (testing "round at various values"
    (is (= 3 (round 2.5)))
    (is (= 2 (round 2.4)))
    (is (= -2 (round -2.4)))
    (is (= -2 (round -2.5)))
    (is (= 0 (round 0.0))))
  (testing "round Float returns Integer"
    (is (= 3 (round (float 2.5))))
    (is (instance? Integer (round (float 2.5))))))

(deftest trunc-test
  (testing "trunc towards zero"
    (is (= 2.0 (trunc 2.9)))
    (is (= -2.0 (trunc -2.9)))
    (is (= 3.0 (trunc 3.1)))
    (is (= -3.0 (trunc -3.1)))
    (is (= 0.0 (trunc 0.5)))
    (is (= 0.0 (trunc -0.5))))
  (testing "trunc Float variant"
    (is (= (float 2.0) (trunc (float 2.9))))
    (is (= (float -2.0) (trunc (float -2.9))))
    (is (instance? Float (trunc (float 2.9))))))

;; ================================================================
;; Hyperbolic functions
;; ================================================================

(deftest sinh-test
  (testing "sinh at known values"
    (is (approx= 0.0 (sinh 0.0)))
    ;; sinh(1) = (e - 1/e) / 2
    (is (approx= (/ (- Math/E (/ 1.0 Math/E)) 2.0) (sinh 1.0))))
  (testing "sinh Float variant"
    (is (approx= 0.0 (sinh (float 0.0)) float-tol))
    (is (instance? Float (sinh (float 0.0))))))

(deftest cosh-test
  (testing "cosh at known values"
    (is (approx= 1.0 (cosh 0.0)))
    ;; cosh(1) = (e + 1/e) / 2
    (is (approx= (/ (+ Math/E (/ 1.0 Math/E)) 2.0) (cosh 1.0))))
  (testing "cosh Float variant"
    (is (approx= 1.0 (cosh (float 0.0)) float-tol))
    (is (instance? Float (cosh (float 0.0))))))

(deftest tanh-test
  (testing "tanh at known values"
    (is (approx= 0.0 (tanh 0.0)))
    ;; tanh(x) approaches +/-1 for large |x|
    (is (approx= 1.0 (tanh 20.0) 1e-8))
    (is (approx= -1.0 (tanh -20.0) 1e-8)))
  (testing "tanh Float variant"
    (is (approx= 0.0 (tanh (float 0.0)) float-tol))
    (is (instance? Float (tanh (float 0.0))))))

;; ================================================================
;; Inverse hyperbolic functions
;; ================================================================

(deftest asinh-test
  (testing "asinh at known values"
    (is (approx= 0.0 (asinh 0.0))))
  (testing "sinh/asinh roundtrip"
    (doseq [x [0.0 1.0 -1.0 5.0 -5.0]]
      (is (approx= x (sinh (asinh x))))))
  (testing "asinh Float variant"
    (is (approx= 0.0 (asinh (float 0.0)) float-tol))
    (is (instance? Float (asinh (float 0.0))))))

(deftest acosh-test
  (testing "acosh at known values"
    (is (approx= 0.0 (acosh 1.0))))
  (testing "cosh/acosh roundtrip for x >= 1"
    (doseq [x [1.0 2.0 5.0 10.0]]
      (is (approx= x (cosh (acosh x))))))
  (testing "acosh Float variant"
    (is (approx= 0.0 (acosh (float 1.0)) float-tol))
    (is (instance? Float (acosh (float 1.0))))))

(deftest atanh-test
  (testing "atanh at known values"
    (is (approx= 0.0 (atanh 0.0))))
  (testing "tanh/atanh roundtrip for |x| < 1"
    (doseq [x [0.0 0.5 -0.5 0.9 -0.9]]
      (is (approx= x (tanh (atanh x))))))
  (testing "atanh Float variant"
    (is (approx= 0.0 (atanh (float 0.0)) float-tol))
    (is (instance? Float (atanh (float 0.0))))))

;; ================================================================
;; Degree/radian conversion
;; ================================================================

(deftest deg2rad-test
  (testing "deg2rad at known values"
    (is (approx= 0.0 (deg2rad 0.0)))
    (is (approx= Math/PI (deg2rad 180.0)))
    (is (approx= (/ Math/PI 2.0) (deg2rad 90.0)))
    (is (approx= (* 2.0 Math/PI) (deg2rad 360.0))))
  (testing "deg2rad Float variant"
    (is (instance? Float (deg2rad (float 90.0))))))

(deftest rad2deg-test
  (testing "rad2deg at known values"
    (is (approx= 0.0 (rad2deg 0.0)))
    (is (approx= 180.0 (rad2deg Math/PI)))
    (is (approx= 90.0 (rad2deg (/ Math/PI 2.0))))
    (is (approx= 360.0 (rad2deg (* 2.0 Math/PI)))))
  (testing "rad2deg Float variant"
    (is (instance? Float (rad2deg (float (/ Math/PI 2.0)))))))

(deftest deg-rad-roundtrip-test
  (testing "deg2rad/rad2deg roundtrip"
    (doseq [d [0.0 45.0 90.0 180.0 270.0 360.0]]
      (is (approx= d (rad2deg (deg2rad d)))))))

;; ================================================================
;; Other math functions
;; ================================================================

(deftest hypot-test
  (testing "hypot at known values"
    (is (approx= 5.0 (hypot 3.0 4.0)))
    (is (approx= 13.0 (hypot 5.0 12.0)))
    (is (approx= 0.0 (hypot 0.0 0.0))))
  (testing "hypot Float variant"
    (is (approx= 5.0 (hypot (float 3.0) (float 4.0)) float-tol))
    (is (instance? Float (hypot (float 3.0) (float 4.0))))))

(deftest cbrt-test
  (testing "cbrt at known values"
    (is (approx= 0.0 (cbrt 0.0)))
    (is (approx= 1.0 (cbrt 1.0)))
    (is (approx= 2.0 (cbrt 8.0)))
    (is (approx= 3.0 (cbrt 27.0)))
    (is (approx= -2.0 (cbrt -8.0))))
  (testing "cbrt Float variant"
    (is (approx= 2.0 (cbrt (float 8.0)) float-tol))
    (is (instance? Float (cbrt (float 8.0))))))

(deftest sqrt-edge-case-test
  (testing "sqrt(-1) is NaN"
    (is (Double/isNaN (Math/sqrt -1.0)))))

;; ================================================================
;; Floating-point special values
;; ================================================================

(deftest nextfloat-prevfloat-test
  (testing "nextfloat is greater"
    (is (> (nextfloat 1.0) 1.0)))
  (testing "prevfloat is smaller"
    (is (< (prevfloat 1.0) 1.0)))
  (testing "nextfloat/prevfloat Float variant"
    (is (> (nextfloat (float 1.0)) (float 1.0)))
    (is (< (prevfloat (float 1.0)) (float 1.0)))
    (is (instance? Float (nextfloat (float 1.0))))
    (is (instance? Float (prevfloat (float 1.0))))))

(deftest eps-test
  (testing "eps at 1.0 equals machine epsilon"
    (is (approx= (Math/ulp 1.0) (eps 1.0))))
  (testing "eps Float variant"
    (is (instance? Float (eps (float 1.0))))))

;; ================================================================
;; Clamping
;; ================================================================

(deftest clamp-double-test
  (testing "clamp Double"
    (is (= 0.0 (clamp -1.0 0.0 1.0)))
    (is (= 0.5 (clamp 0.5 0.0 1.0)))
    (is (= 1.0 (clamp 2.0 0.0 1.0)))))

(deftest clamp-float-test
  (testing "clamp Float"
    (is (= (float 0.0) (clamp (float -1.0) (float 0.0) (float 1.0))))
    (is (= (float 0.5) (clamp (float 0.5) (float 0.0) (float 1.0))))
    (is (= (float 1.0) (clamp (float 2.0) (float 0.0) (float 1.0))))))

(deftest clamp-long-test
  (testing "clamp Long"
    (is (= 5 (clamp 3 5 10)))
    (is (= 7 (clamp 7 5 10)))
    (is (= 10 (clamp 15 5 10)))))

;; ================================================================
;; fma (fused multiply-add)
;; ================================================================

(deftest fma-test
  (testing "fma Double: x*y + z"
    (is (approx= 7.0 (fma 2.0 3.0 1.0)))
    (is (approx= 0.0 (fma 0.0 5.0 0.0)))
    (is (approx= -1.0 (fma 2.0 3.0 -7.0))))
  (testing "fma Float"
    (is (approx= 7.0 (fma (float 2.0) (float 3.0) (float 1.0)) float-tol))
    (is (instance? Float (fma (float 2.0) (float 3.0) (float 1.0))))))

;; ================================================================
;; copysign
;; ================================================================

(deftest copysign-test
  (testing "copysign Double"
    (is (= 3.0 (copysign 3.0 1.0)))
    (is (= -3.0 (copysign 3.0 -1.0)))
    (is (= 3.0 (copysign -3.0 1.0)))
    (is (= -3.0 (copysign -3.0 -1.0))))
  (testing "copysign with zero sign"
    (is (= 5.0 (copysign 5.0 0.0))))
  (testing "copysign Float"
    (is (= (float 3.0) (copysign (float 3.0) (float 1.0))))
    (is (= (float -3.0) (copysign (float 3.0) (float -1.0))))
    (is (instance? Float (copysign (float 3.0) (float 1.0))))))

;; ================================================================
;; flipsign
;; ================================================================

(deftest flipsign-test
  (testing "flipsign Double: flip if y negative"
    (is (= 3.0 (flipsign 3.0 1.0)))
    (is (= -3.0 (flipsign 3.0 -1.0)))
    (is (= -3.0 (flipsign -3.0 1.0)))
    (is (= 3.0 (flipsign -3.0 -1.0))))
  (testing "flipsign with y=0 (not negative, so no flip)"
    (is (= 5.0 (flipsign 5.0 0.0))))
  (testing "flipsign Float"
    (is (approx= 3.0 (flipsign (float 3.0) (float 1.0)) float-tol))
    (is (approx= -3.0 (flipsign (float 3.0) (float -1.0)) float-tol))
    (is (instance? Float (flipsign (float 3.0) (float 1.0))))))

;; ================================================================
;; sincos
;; ================================================================

(deftest sincos-test
  (testing "sincos returns [sin cos]"
    (let [[s c] (sincos 0.0)]
      (is (approx= 0.0 s))
      (is (approx= 1.0 c)))
    (let [[s c] (sincos (/ Math/PI 2.0))]
      (is (approx= 1.0 s))
      (is (approx= 0.0 c)))))

;; ================================================================
;; signum
;; ================================================================

(deftest signum-test
  (testing "signum Double"
    (is (= 1.0 (signum 42.0)))
    (is (= -1.0 (signum -42.0)))
    (is (= 0.0 (signum 0.0))))
  (testing "signum Float"
    (is (= (float 1.0) (signum (float 42.0))))
    (is (= (float -1.0) (signum (float -42.0))))
    (is (= (float 0.0) (signum (float 0.0))))
    (is (instance? Float (signum (float 1.0))))))
