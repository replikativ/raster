(ns raster.ad.jet
  "Jet type for efficient higher-order derivatives via truncated Taylor series.

   A Jet stores scaled Taylor coefficients [f(a), f'(a)/1!, ..., f^(k)(a)/k!]
   forming a truncated polynomial ring. Arithmetic follows convolution rules,
   math functions use standard recurrences (Griewank & Walther, Algorithm 13.1).

   Jets vs nested Duals:
     Nested Duals: O(2^k) storage/work for k-th order
     Jets:         O(k) storage, O(k²) work per operation

   Usage:
     (require '[raster.ad.jet :as jet])
     (jet/higher-derivatives (fn [x] (* (sin x) x)) 1.0 4)
     ;; => all derivatives up to 4th order at x=1"
  (:refer-clojure :exclude [+ - * / abs aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue defval]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.numeric]
            [raster.math]
            [raster.types.promote :as promote]))

;; ================================================================
;; Jet value type — truncated Taylor coefficients
;; ================================================================

(defvalue Jet [order :- Long, coeffs :- (Array double)])
;; coeffs has (inc order) elements
;; coeffs[i] = f^(i)(a) / i!

;; ================================================================
;; Constructors
;; ================================================================

(defn jet
  "Create an identity jet at value a with given order.
   Represents the identity function x at x=a:
   coeffs = [a, 1, 0, 0, ...]"
  [^double a ^long order]
  (let [c (double-array (inc order))]
    (aset c 0 a)
    (when (>= order 1)
      (aset c 1 1.0))
    (->Jet order c)))

(defn jet-const
  "Create a constant jet at value a with given order.
   coeffs = [a, 0, 0, ...]"
  [^double a ^long order]
  (let [c (double-array (inc order))]
    (aset c 0 a)
    (->Jet order c)))

(defn jet-from-coeffs
  "Create a jet from explicit coefficients array."
  [^doubles coeffs]
  (->Jet (dec (alength coeffs)) coeffs))

;; ================================================================
;; Extraction
;; ================================================================

(defn taylor-coefficient
  "Extract the k-th scaled Taylor coefficient: f^(k)(a) / k!"
  ^double [^Jet j ^long k]
  (aget (.coeffs j) k))

(defn taylor-derivative
  "Extract the k-th derivative at the expansion point.
   Unscales: result = coeffs[k] * k!"
  ^double [^Jet j ^long k]
  (let [c (aget (.coeffs j) k)]
    (loop [i 1 fac 1.0]
      (if (> i k)
        (clojure.core/* c fac)
        (recur (inc i) (clojure.core/* fac (double i)))))))

(defn primal
  "Extract the primal value (0th coefficient)."
  ^double [^Jet j]
  (aget (.coeffs j) 0))

;; ================================================================
;; Internal helpers
;; ================================================================

(defn- ensure-same-order ^long [^Jet a ^Jet b]
  (let [oa (.order a) ob (.order b)]
    (when (not= oa ob)
      (throw (ex-info "Jet order mismatch" {:a oa :b ob})))
    oa))

;; ================================================================
;; Arithmetic: Jet × Jet (truncated polynomial ring)
;; ================================================================

(deftm raster.numeric/+ [x :- Jet, y :- Jet] :- Jet
  (let [n (ensure-same-order x y)
        cx (.coeffs x) cy (.coeffs y)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/+ (aget cx i) (aget cy i))))
    (->Jet n out)))

(deftm raster.numeric/- [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/- (aget cx i))))
    (->Jet n out)))

(deftm raster.numeric/- [x :- Jet, y :- Jet] :- Jet
  (let [n (ensure-same-order x y)
        cx (.coeffs x) cy (.coeffs y)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/- (aget cx i) (aget cy i))))
    (->Jet n out)))

;; Multiplication: truncated Cauchy product
;; c[k] = Σ_{j=0}^{k} a[j] * b[k-j]
(deftm raster.numeric/* [x :- Jet, y :- Jet] :- Jet
  (let [n (ensure-same-order x y)
        cx (.coeffs x) cy (.coeffs y)
        out (double-array (inc n))]
    (dotimes [k (inc n)]
      (let [s (loop [j 0 acc 0.0]
                (if (> j k)
                  acc
                  (recur (inc j) (clojure.core/+ acc (clojure.core/* (aget cx j)
                                                                     (aget cy (clojure.core/- k j)))))))]
        (aset out k s)))
    (->Jet n out)))

;; Division: recurrent solve from a = b * c → c[k] = (a[k] - Σ_{j=1}^{k} c[k-j]*b[j]) / b[0]
(deftm raster.numeric// [x :- Jet, y :- Jet] :- Jet
  (let [n (ensure-same-order x y)
        cx (.coeffs x) cy (.coeffs y)
        b0 (aget cy 0)
        out (double-array (inc n))]
    (aset out 0 (clojure.core// (aget cx 0) b0))
    (dotimes [k* n]
      (let [k (inc k*)
            s (loop [j 1 acc 0.0]
                (if (> j k)
                  acc
                  (recur (inc j) (clojure.core/+ acc (clojure.core/* (aget out (clojure.core/- k j))
                                                                     (aget cy j))))))]
        (aset out k (clojure.core// (clojure.core/- (aget cx k) s) b0))))
    (->Jet n out)))

;; ================================================================
;; Arithmetic: Jet × Number and Number × Jet
;; ================================================================

(deftm raster.numeric/+ [x :- Jet, y :- Number] :- Jet
  (let [c (aclone (.coeffs x))]
    (aset c 0 (clojure.core/+ (aget c 0) (double y)))
    (->Jet (.order x) c)))

(deftm raster.numeric/+ [x :- Number, y :- Jet] :- Jet
  (let [c (aclone (.coeffs y))]
    (aset c 0 (clojure.core/+ (double x) (aget c 0)))
    (->Jet (.order y) c)))

(deftm raster.numeric/- [x :- Jet, y :- Number] :- Jet
  (let [c (aclone (.coeffs x))]
    (aset c 0 (clojure.core/- (aget c 0) (double y)))
    (->Jet (.order x) c)))

(deftm raster.numeric/- [x :- Number, y :- Jet] :- Jet
  (let [n (.order y) cy (.coeffs y)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/- (aget cy i))))
    (aset out 0 (clojure.core/- (double x) (aget cy 0)))
    (->Jet n out)))

(deftm raster.numeric/* [x :- Jet, y :- Number] :- Jet
  (let [y (double y)
        n (.order x) cx (.coeffs x)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/* (aget cx i) y)))
    (->Jet n out)))

(deftm raster.numeric/* [x :- Number, y :- Jet] :- Jet
  (let [x (double x)
        n (.order y) cy (.coeffs y)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core/* x (aget cy i))))
    (->Jet n out)))

(deftm raster.numeric// [x :- Jet, y :- Number] :- Jet
  (let [y (double y)
        n (.order x) cx (.coeffs x)
        out (double-array (inc n))]
    (dotimes [i (inc n)]
      (aset out i (clojure.core// (aget cx i) y)))
    (->Jet n out)))

(deftm raster.numeric// [x :- Number, y :- Jet] :- Jet
  ;; x / y where y is Jet: treat as x * (1/y)
  ;; c[0] = x/b[0], c[k] = -(1/b[0]) * Σ_{j=1}^{k} c[k-j]*b[j]
  (let [x (double x)
        n (.order y) cy (.coeffs y)
        b0 (aget cy 0)
        out (double-array (inc n))]
    (aset out 0 (clojure.core// x b0))
    (dotimes [k* n]
      (let [k (inc k*)
            s (loop [j 1 acc 0.0]
                (if (> j k)
                  acc
                  (recur (inc j) (clojure.core/+ acc (clojure.core/* (aget out (clojure.core/- k j))
                                                                     (aget cy j))))))]
        (aset out k (clojure.core// (clojure.core/- s) b0))))
    (->Jet n out)))

;; ================================================================
;; Math functions via standard recurrences
;; (Griewank & Walther, "Evaluating Derivatives", Algorithm 13.1)
;; ================================================================

;; exp(f): e[0]=exp(f[0]), e[k] = (1/k) * Σ_{j=1}^{k} j * f[j] * e[k-j]
(deftm raster.math/exp [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        out (double-array (inc n))]
    (aset out 0 (Math/exp (aget cx 0)))
    (dotimes [k* n]
      (let [k (inc k*)
            s (loop [j 1 acc 0.0]
                (if (> j k)
                  acc
                  (recur (inc j)
                         (clojure.core/+ acc (clojure.core/* (double j)
                                                             (clojure.core/* (aget cx j)
                                                                             (aget out (clojure.core/- k j))))))))]
        (aset out k (clojure.core// s (double k)))))
    (->Jet n out)))

;; log(f): l[0]=log(f[0]), l[k] = (1/f[0]) * (f[k] - (1/k) * Σ_{j=1}^{k-1} j * l[j] * f[k-j])
(deftm raster.math/log [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        f0 (aget cx 0)
        inv-f0 (clojure.core// 1.0 f0)
        out (double-array (inc n))]
    (aset out 0 (Math/log f0))
    (dotimes [k* n]
      (let [k (inc k*)
            s (loop [j 1 acc 0.0]
                (if (>= j k)
                  acc
                  (recur (inc j)
                         (clojure.core/+ acc (clojure.core/* (double j)
                                                             (clojure.core/* (aget out j)
                                                                             (aget cx (clojure.core/- k j))))))))]
        (aset out k (clojure.core/* inv-f0
                                    (clojure.core/- (aget cx k)
                                                    (clojure.core// s (double k)))))))
    (->Jet n out)))

;; sin/cos: coupled recurrence
;; s[0]=sin(f[0]), c[0]=cos(f[0])
;; s[k] = (1/k) * Σ_{j=1}^{k} j * f[j] * c[k-j]
;; c[k] = -(1/k) * Σ_{j=1}^{k} j * f[j] * s[k-j]
(deftm raster.math/sin [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        s (double-array (inc n))
        c (double-array (inc n))]
    (aset s 0 (Math/sin (aget cx 0)))
    (aset c 0 (Math/cos (aget cx 0)))
    (dotimes [k* n]
      (let [k (inc k*)
            ;; s[k] = (1/k) * Σ_{j=1}^{k} j * f[j] * c[k-j]
            ss (loop [j 1 acc 0.0]
                 (if (> j k) acc
                     (recur (inc j)
                            (clojure.core/+ acc (clojure.core/* (double j)
                                                                (clojure.core/* (aget cx j)
                                                                                (aget c (clojure.core/- k j))))))))
            ;; c[k] = -(1/k) * Σ_{j=1}^{k} j * f[j] * s[k-j]
            cs (loop [j 1 acc 0.0]
                 (if (> j k) acc
                     (recur (inc j)
                            (clojure.core/+ acc (clojure.core/* (double j)
                                                                (clojure.core/* (aget cx j)
                                                                                (aget s (clojure.core/- k j))))))))]
        (aset s k (clojure.core// ss (double k)))
        (aset c k (clojure.core/- (clojure.core// cs (double k))))))
    (->Jet n s)))

(deftm raster.math/cos [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        s (double-array (inc n))
        c (double-array (inc n))]
    (aset s 0 (Math/sin (aget cx 0)))
    (aset c 0 (Math/cos (aget cx 0)))
    (dotimes [k* n]
      (let [k (inc k*)
            ss (loop [j 1 acc 0.0]
                 (if (> j k) acc
                     (recur (inc j)
                            (clojure.core/+ acc (clojure.core/* (double j)
                                                                (clojure.core/* (aget cx j)
                                                                                (aget c (clojure.core/- k j))))))))
            cs (loop [j 1 acc 0.0]
                 (if (> j k) acc
                     (recur (inc j)
                            (clojure.core/+ acc (clojure.core/* (double j)
                                                                (clojure.core/* (aget cx j)
                                                                                (aget s (clojure.core/- k j))))))))]
        (aset s k (clojure.core// ss (double k)))
        (aset c k (clojure.core/- (clojure.core// cs (double k))))))
    (->Jet n c)))

;; tan: t[0]=tan(f[0]), via t = sin/cos composition
;; More efficient: t[k] = s[k] + (1/k) * Σ_{j=1}^{k} j * f[j] * t²[k-j]
;; But for simplicity, use sin/cos jets then divide
(deftm raster.math/tan [x :- Jet] :- Jet
  (raster.numeric// (raster.math/sin x) (raster.math/cos x)))

;; sqrt(f): via s[0]=sqrt(f[0]), s[k] = (1/(2*s[0])) * (f[k] - (1/k) * Σ_{j=1}^{k-1} ...)
;; Recurrence: s[0]=sqrt(f[0])
;; s[k] = (f[k] - Σ_{j=1}^{k-1} s[j]*s[k-j]) / (2*s[0])
(deftm raster.numeric/sqrt [x :- Jet] :- Jet
  (let [n (.order x) cx (.coeffs x)
        out (double-array (inc n))
        s0 (Math/sqrt (aget cx 0))
        inv2s0 (clojure.core// 1.0 (clojure.core/* 2.0 s0))]
    (aset out 0 s0)
    (dotimes [k* n]
      (let [k (inc k*)
            s (loop [j 1 acc 0.0]
                (if (>= j k) acc
                    (recur (inc j) (clojure.core/+ acc (clojure.core/* (aget out j)
                                                                       (aget out (clojure.core/- k j)))))))]
        (aset out k (clojure.core/* (clojure.core/- (aget cx k) s) inv2s0))))
    (->Jet n out)))

;; pow(f, n): via exp(n * log(f))
(deftm raster.numeric/pow [x :- Jet, n :- Double] :- Jet
  (raster.math/exp (raster.numeric/* n (raster.math/log x))))

;; abs: subdifferential at value
(deftm raster.numeric/abs [x :- Jet] :- Jet
  (let [v (aget (.coeffs x) 0)]
    (if (>= v 0.0)
      x
      (raster.numeric/- x))))

;; sign: zero derivatives
(deftm raster.numeric/sign [x :- Jet] :- Jet
  (jet-const (Math/signum (aget (.coeffs x) 0)) (.order x)))

;; zero? / zero / one for numeric tower compatibility
(deftm raster.numeric/zero? [x :- Jet]
  (== 0.0 (aget (.coeffs x) 0)))

(deftm raster.numeric/zero [x :- Jet] :- Jet
  (jet-const 0.0 (.order x)))

(deftm raster.numeric/one [x :- Jet] :- Jet
  (jet-const 1.0 (.order x)))

(deftm raster.numeric/real-value [x :- Jet] :- Double
  (primal x))

(deftm raster.numeric/derivative-value [x :- Jet] :- Double
  (if (>= (.order x) 1)
    (taylor-derivative x 1)
    0.0))

;; ================================================================
;; Comparisons (dispatch on primal value)
;; ================================================================

(deftm raster.numeric/< [x :- Jet, y :- Jet]
  (clojure.core/< (aget (.coeffs x) 0) (aget (.coeffs y) 0)))

(deftm raster.numeric/<= [x :- Jet, y :- Jet]
  (clojure.core/<= (aget (.coeffs x) 0) (aget (.coeffs y) 0)))

(deftm raster.numeric/> [x :- Jet, y :- Jet]
  (clojure.core/> (aget (.coeffs x) 0) (aget (.coeffs y) 0)))

(deftm raster.numeric/>= [x :- Jet, y :- Jet]
  (clojure.core/>= (aget (.coeffs x) 0) (aget (.coeffs y) 0)))

(deftm raster.numeric/== [x :- Jet, y :- Jet]
  (clojure.core/== (aget (.coeffs x) 0) (aget (.coeffs y) 0)))

;; ================================================================
;; Type promotion
;; ================================================================

(defval TJet)
(promote/register-type-tag! Jet (->TJet))
(promote/register-promote-rule! Long Jet Jet)
(promote/register-promote-rule! Double Jet Jet)
(promote/register-promote-rule! Float Jet Jet)
(promote/register-promote-rule! Integer Jet Jet)

(deftm raster.types.promote/convert [t :- TJet, x :- Jet] :- Jet x)
(deftm raster.types.promote/convert [t :- TJet, x :- Number] :- Jet
  (jet-const (.doubleValue ^Number x) 1))

;; ================================================================
;; Public API
;; ================================================================

(defn higher-derivatives
  "Compute f, f', f'', ..., f^(k) at point a in a single pass.
   Returns a double-array of length (inc k) with [f(a), f'(a), ..., f^(k)(a)].

   Uses jet propagation: O(k²) work instead of O(2^k) for nested Duals."
  ^doubles [f ^double a ^long k]
  (let [j (f (jet a k))
        result (double-array (inc k))]
    (if (instance? Jet j)
      (dotimes [i (inc k)]
        (aset result i (taylor-derivative ^Jet j i)))
      ;; f returned a constant (not Jet)
      (aset result 0 (double j)))
    result))

(defn jet-derivative
  "Compute the k-th derivative of f at a using Jet propagation.
   More efficient than k nested forward-AD passes."
  ^double [f ^double a ^long k]
  (let [j (f (jet a k))]
    (if (instance? Jet j)
      (taylor-derivative ^Jet j k)
      (if (zero? k) (double j) 0.0))))
