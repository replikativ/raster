(ns raster.sci.quadrature
  "Numerical integration (quadrature) algorithms.

  Provides adaptive Gauss-Kronrod quadrature, trapezoidal rule,
  and Simpson's rule following Julia's QuadGK.jl patterns.

  Usage:
    (require '[raster.sci.quadrature :refer [quadgk trapz simps]])
    (quadgk (fn [x] (* x x)) 0.0 1.0)         ;=> [0.3333... ~0.0]
    (trapz (double-array [0 1 4 9]) 1.0)        ;=> 9.5
    (simps (fn [x] (m/sin x)) 0.0 n/pi)   ;=> ~2.0"
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n :refer [+ - * /]]))

;; ================================================================
;; Gauss-Kronrod 7-15 point rule
;; ================================================================

;; Gauss-Kronrod nodes and weights for the 7-point Gauss / 15-point Kronrod rule
;; on [-1, 1]. The 7 Gauss nodes are a subset of the 15 Kronrod nodes.

;; 15-point Kronrod nodes (positive half, symmetric about 0)
(def ^:private kronrod-nodes
  (double-array [0.0
                 0.20778495500789846760
                 0.40584515137739716691
                 0.58608723546769113507
                 0.74153118559939443986
                 0.86486442335976907279
                 0.94910791234275852453
                 0.99145537112081263921]))

;; 15-point Kronrod weights
(def ^:private kronrod-weights
  (double-array [0.20948214108472782801   ;; node 0
                 0.20443294007529889241   ;; node 1
                 0.19035057806478540991   ;; node 2
                 0.16900472663926790283   ;; node 3
                 0.14065325971552591875   ;; node 4
                 0.10479001032225018384   ;; node 5
                 0.06309209262997855329   ;; node 6
                 0.02293532201052922497]));; node 7

;; 7-point Gauss weights (for nodes at Kronrod indices 0, 2, 4, 6)
(def ^:private gauss-weights
  (double-array [0.41795918367346938776   ;; node 0 (center)
                 0.38183005050511894495   ;; node 2
                 0.27970539148927666790   ;; node 4
                 0.12948496616886969327]));; node 6

(declare adaptive-quadgk)

(deftm gk15 [f :- Object, a :- Double, b :- Double]
  (let [mid (* 0.5 (+ a b))
        half-len (* 0.5 (- b a))
        f-center (double (f mid))
        kronrod-sum (* (aget kronrod-weights 0) f-center)
        gauss-sum (* (aget gauss-weights 0) f-center)]
    (loop [j 1
           k-sum kronrod-sum
           g-sum gauss-sum]
      (if (>= j 8)
        [(* half-len k-sum) (* half-len g-sum)]
        (let [node (aget kronrod-nodes j)
              x-plus (+ mid (* half-len node))
              x-minus (- mid (* half-len node))
              f-plus (double (f x-plus))
              f-minus (double (f x-minus))
              f-sym (+ f-plus f-minus)
              kw (aget kronrod-weights j)
              k-sum (+ k-sum (* kw f-sym))
              g-sum (if (== 0 (bit-and j 1))
                      (+ g-sum (* (aget gauss-weights (bit-shift-right j 1)) f-sym))
                      g-sum)]
          (recur (inc j) k-sum g-sum))))))

(deftm adaptive-quadgk [f :- Object, a :- Double, b :- Double, atol :- Double, rtol :- Double, max-depth :- Long, depth :- Long]
  (let [[k-result g-result] (gk15 f a b)
        err (n/abs (- (double k-result) (double g-result)))]
    (if (or (>= depth max-depth)
            (<= err (n/max atol (* rtol (n/abs (double k-result))))))
      [k-result err]
      (let [mid (* 0.5 (+ a b))
            [left-val left-err] (adaptive-quadgk f a mid atol rtol max-depth (inc depth))
            [right-val right-err] (adaptive-quadgk f mid b atol rtol max-depth (inc depth))]
        [(+ (double left-val) (double right-val)) (+ (double left-err) (double right-err))]))))

(deftm quadgk [f :- Object, a :- Double, b :- Double, atol :- Double, rtol :- Double, max-depth :- Long]
  (adaptive-quadgk f a b atol rtol max-depth 0))

(deftm quadgk [f :- Object, a :- Double, b :- Double]
  (quadgk f a b 1e-12 1e-8 15))

;; ================================================================
;; Trapezoidal rule (for sampled data)
;; ================================================================

;; INPUT-GENERIC: accept (Array T), accumulate and return Double
(deftm trapz (All [T] [ys :- (Array T), dx :- Double] :- Double
                  (let [n (alength ys)]
                    (if (< n 2)
                      0.0
                      (let [n-1 (dec n)]
                        (* dx (+ (* 0.5 (+ (double (aget ys 0)) (double (aget ys n-1))))
                                 (loop [i 1 s 0.0]
                                   (if (>= i n-1)
                                     s
                                     (recur (inc i) (+ s (double (aget ys i)))))))))))))

(deftm trapz-xy (All [T] [xs :- (Array T), ys :- (Array T)] :- Double
                     (let [n (alength xs)]
                       (if (< n 2)
                         0.0
                         (loop [i 0 s 0.0]
                           (if (>= i (dec n))
                             s
                             (let [dx (- (double (aget xs (inc i))) (double (aget xs i)))
                                   avg (* 0.5 (+ (double (aget ys i)) (double (aget ys (inc i)))))]
                               (recur (inc i) (+ s (* dx avg))))))))))

;; ================================================================
;; Simpson's rule
;; ================================================================

(deftm simps [f :- Object, a :- Double, b :- Double, n :- Long] :- Double
  (let [n (if (odd? n) (inc n) n)
        h (/ (- b a) (double n))
        fa (double (f a))
        fb (double (f b))]
    (let [interior (loop [i 1 s 0.0]
                     (if (>= i n)
                       s
                       (let [xi (+ a (* (double i) h))
                             fi (double (f xi))
                             coeff (if (even? i) 2.0 4.0)]
                         (recur (inc i) (+ s (* coeff fi))))))]
      (* (/ h 3.0) (+ fa fb interior)))))

(deftm simps [f :- Object, a :- Double, b :- Double] :- Double
  (simps f a b 100))
