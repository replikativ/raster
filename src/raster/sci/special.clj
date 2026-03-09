(ns raster.sci.special
  "Special mathematical functions as deftm generic functions.

  Provides typed dispatch for gamma, beta, error, Bessel, and other
  special functions. All implementations use standard numerical recipes
  (Lanczos, Cephes-style rational polynomials, asymptotic expansions).

  Usage:
    (require '[raster.sci.special :refer [lgamma gamma digamma erf besselj0]])
    (lgamma 5.0)       ;; => 3.178053830...
    (besselj0 1.0)     ;; => 0.765197686..."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; Constants
;; ================================================================

(def ^:private ^:const INV_SQRT_PI (/ 2.0 (n/sqrt n/pi)))
(def ^:private ^:const TWO_OVER_PI (/ 2.0 n/pi))
(def ^:private ^:const SQ2OPI (n/sqrt TWO_OVER_PI))

;; ================================================================
;; Gamma family
;; ================================================================

(declare lgamma digamma trigamma betainc)

(deftm lgamma [x :- Double] :- Double
  (if (< x 0.5)
    (- (m/log (/ n/pi (m/sin (* n/pi x))))
       (lgamma (- 1.0 x)))
    (let [x (- x 1.0)
          coeffs (double-array [676.5203681218851
                                -1259.1392167224028
                                771.32342877765313
                                -176.61502916214059
                                12.507343278686905
                                -0.13857109526572012
                                9.9843695780195716e-6
                                1.5056327351493116e-7])
          g 7.0
          t (+ x g 0.5)
          s (loop [i 7 acc 0.99999999999980993]
              (if (neg? i)
                acc
                (recur (dec i) (+ acc (/ (aget coeffs i) (+ x (double i) 1.0))))))]
      (+ (* 0.5 (m/log (* 2.0 n/pi)))
         (* (+ x 0.5) (m/log t))
         (- t)
         (m/log s)))))

(deftm gamma [x :- Double] :- Double
  (m/exp (lgamma x)))

(deftm digamma [x :- Double] :- Double
  (if (< x 1e-6)
    (- (- (/ 1.0 x)) 0.5772156649015329)
    (if (< x 6.0)
      (loop [xx x acc 0.0]
        (if (>= xx 6.0)
          (+ acc (digamma xx))
          (recur (+ xx 1.0) (- acc (/ 1.0 xx)))))
      (let [inv-x (/ 1.0 x)
            inv-x2 (* inv-x inv-x)]
        (- (m/log x)
           (* 0.5 inv-x)
           (* inv-x2
              (- (/ 1.0 12.0)
                 (* inv-x2
                    (- (/ 1.0 120.0)
                       (* inv-x2
                          (- (/ 1.0 252.0)
                             (* inv-x2
                                (- (/ 1.0 240.0)
                                   (* inv-x2
                                      (/ 1.0 132.0)))))))))))))))

(deftm trigamma [x :- Double] :- Double
  (if (< x 1e-6)
    (/ 1.0 (* x x))
    (if (< x 6.0)
      (loop [xx x acc 0.0]
        (if (>= xx 6.0)
          (+ acc (trigamma xx))
          (recur (+ xx 1.0) (+ acc (/ 1.0 (* xx xx))))))
      (let [inv-x (/ 1.0 x)
            inv-x2 (* inv-x inv-x)]
        (* inv-x
           (+ 1.0
              (* inv-x
                 (+ 0.5
                    (* inv-x
                       (+ (/ 1.0 6.0)
                          (* inv-x2
                             (+ (/ -1.0 30.0)
                                (* inv-x2
                                   (+ (/ 1.0 42.0)
                                      (* inv-x2
                                         (/ -1.0 30.0))))))))))))))))

(deftm gammainc [a :- Double, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (let [lga (lgamma a)]
      (loop [n 0 sum 0.0 term (/ 1.0 a)]
        (if (or (> n 200) (< (n/abs term) (* 1e-14 (n/abs sum))))
          (* sum (m/exp (- (* a (m/log x)) x lga)))
          (let [new-sum (+ sum term)
                new-term (* term (/ x (+ a (double n) 1.0)))]
            (recur (inc n) new-sum new-term)))))))

(deftm gammaincc [a :- Double, x :- Double] :- Double
  (- 1.0 (gammainc a x)))

;; ================================================================
;; Beta family
;; ================================================================

(deftm lbeta [a :- Double, b :- Double] :- Double
  (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

(deftm beta [a :- Double, b :- Double] :- Double
  (m/exp (lbeta a b)))

(deftm betainc [a :- Double, b :- Double, x :- Double] :- Double
  (let [max-iter 200
        eps 3.0e-12
        fpmin 1.0e-30]
    (cond
      (<= x 0.0) 0.0
      (>= x 1.0) 1.0
      (> x (/ (+ a 1.0) (+ a b 2.0)))
      (- 1.0 (betainc b a (- 1.0 x)))
      :else
      (let [lnpfx (- (+ (* a (m/log x)) (* b (m/log (- 1.0 x))))
                     (lbeta a b) (m/log a))
            qab (+ a b)
            qap (+ a 1.0)
            qam (- a 1.0)
            c0 1.0
            d0 (let [v (- 1.0 (/ (* qab x) qap))]
                 (if (< (n/abs v) fpmin) (/ 1.0 fpmin) (/ 1.0 v)))
            h0 d0]
        (* (m/exp lnpfx)
           (loop [m 1 c c0 d d0 h h0]
             (if (>= m max-iter)
               h
               (let [dm (double m)
                     em (/ (* dm (- b dm) x)
                           (* (+ qam (* 2.0 dm)) (+ a (* 2.0 dm))))
                     d1 (+ 1.0 (* em d))
                     d1 (if (< (n/abs d1) fpmin) fpmin d1)
                     d1 (/ 1.0 d1)
                     c1 (+ 1.0 (/ em c))
                     c1 (if (< (n/abs c1) fpmin) fpmin c1)
                     h (* h d1 c1)
                     ep (/ (- (* (+ a dm) (+ qab dm) x))
                           (* (+ a (* 2.0 dm)) (+ qap (* 2.0 dm))))
                     d2 (+ 1.0 (* ep d1))
                     d2 (if (< (n/abs d2) fpmin) fpmin d2)
                     d2 (/ 1.0 d2)
                     c2 (+ 1.0 (/ ep c1))
                     c2 (if (< (n/abs c2) fpmin) fpmin c2)
                     del (* d2 c2)
                     h (* h del)]
                 (if (< (n/abs (- del 1.0)) eps)
                   h
                   (recur (inc m) c2 d2 h))))))))))

(deftm betaincinv [a :- Double, b :- Double, p :- Double] :- Double
  (cond
    (<= p 0.0) 0.0
    (>= p 1.0) 1.0
    :else
    (let [lna (m/log (/ a (+ a b)))
          lnb (m/log (/ b (+ a b)))
          t (m/exp (* a lna))
          u (m/exp (* b lnb))
          w (+ t u)
          x0 (if (< p (/ t w))
               (n/pow (* p w a) (/ 1.0 a))
               (- 1.0 (n/pow (* (- 1.0 p) w b) (/ 1.0 b))))]
      (loop [x (n/max 1e-12 (n/min (- 1.0 1e-12) x0))
             lo 0.0
             hi 1.0
             iter 0]
        (if (>= iter 100)
          x
          (let [bp (betainc a b x)
                err (- bp p)]
            (if (< (n/abs err) 1e-14)
              x
              (let [deriv (m/exp (- (+ (* (- a 1.0) (m/log x))
                                       (* (- b 1.0) (m/log (- 1.0 x))))
                                    (lbeta a b)))
                    x-newton (- x (/ err deriv))
                    lo (if (> err 0.0) lo x)
                    hi (if (> err 0.0) x hi)
                    x-new (if (and (> x-newton lo) (< x-newton hi))
                            x-newton
                            (* 0.5 (+ lo hi)))]
                (recur x-new lo hi (inc iter))))))))))

;; ================================================================
;; Error functions
;; ================================================================

(deftm erf [x :- Double] :- Double
  (let [sign (if (neg? x) -1.0 1.0)
        x (n/abs x)
        t (/ 1.0 (m/fma 0.3275911 x 1.0))
        y (* t (m/fma t (m/fma t (m/fma t (m/fma t 1.061405429
                                                 -1.453152027)
                                        1.421413741)
                               -0.284496736)
                      0.254829592))
        result (- 1.0 (* y (m/exp (- (* x x)))))]
    (* sign result)))

(deftm erfc [x :- Double] :- Double
  (- 1.0 (erf x)))

(deftm erfinv [y :- Double] :- Double
  (if (== y 0.0)
    0.0
    (let [a  0.147
          ln1my2 (m/log (- 1.0 (* y y)))
          b  (+ (/ 2.0 (* n/pi a)) (* 0.5 ln1my2))
          x0 (* (m/signum y)
                (n/sqrt (- (n/sqrt (- (* b b) (/ ln1my2 a))) b)))]
      (loop [x x0 iter 0]
        (if (>= iter 8)
          x
          (let [err (- y (erf x))
                deriv (* INV_SQRT_PI (m/exp (- (* x x))))
                x-new (+ x (/ err deriv))]
            (if (< (n/abs err) 1e-15)
              x-new
              (recur x-new (inc iter)))))))))

(deftm erfcinv [x :- Double] :- Double
  (erfinv (- 1.0 x)))

;; ================================================================
;; Bessel functions -- Cephes-style rational polynomial approximations
;; ================================================================

;; --- besselj0 ---

(deftm besselj0 [x :- Double] :- Double
  (let [ax (n/abs x)]
    (if (<= ax 8.0)
      (let [y (* x x)
            ans1 (+ 57568490574.0
                    (* y (+ -13362590354.0
                            (* y (+ 651619640.7
                                    (* y (+ -11214424.18
                                            (* y (+ 77392.33017
                                                    (* y -184.9052456))))))))))
            ans2 (+ 57568490411.0
                    (* y (+ 1029532985.0
                            (* y (+ 9494680.718
                                    (* y (+ 59272.64853
                                            (* y (+ 267.8532712
                                                    y)))))))))]
        (/ ans1 ans2))
      (let [z (/ 8.0 ax)
            y (* z z)
            xx (- ax (* 0.25 n/pi))
            p0 (+ 1.0
                  (* y (+ -1.098628627e-3
                          (* y (+ 2.734510407e-5
                                  (* y (+ -2.073370639e-6
                                          (* y 2.093887211e-7))))))))
            q0 (* z
                  (+ -0.01562499995
                     (* y (+ 1.430488765e-4
                             (* y (+ -6.911147651e-6
                                     (* y (+ 7.621095161e-7
                                             (* y -9.34945152e-8)))))))))]
        (* SQ2OPI (/ 1.0 (n/sqrt ax))
           (- (* p0 (m/cos xx))
              (* q0 (m/sin xx))))))))

;; --- besselj1 ---

(deftm besselj1 [x :- Double] :- Double
  (let [ax (n/abs x)]
    (if (<= ax 8.0)
      (let [y (* x x)
            ans1 (* x
                    (+ 72362614232.0
                       (* y (+ -7895059235.0
                               (* y (+ 242396853.1
                                       (* y (+ -2972611.439
                                               (* y (+ 15704.48260
                                                       (* y -30.16036606)))))))))))
            ans2 (+ 144725228442.0
                    (* y (+ 2300535178.0
                            (* y (+ 18583304.74
                                    (* y (+ 99447.43394
                                            (* y (+ 376.9991397
                                                    y)))))))))]
        (/ ans1 ans2))
      (let [z (/ 8.0 ax)
            y (* z z)
            xx (- ax (* 0.75 n/pi))
            p1 (+ 1.0
                  (* y (+ 1.83105e-3
                          (* y (+ -3.516396496e-5
                                  (* y (+ 2.457520174e-6
                                          (* y -2.40337019e-7))))))))
            q1 (* z
                  (+ 0.04687499995
                     (* y (+ -2.002690873e-4
                             (* y (+ 8.449199096e-6
                                     (* y (+ -8.8228987e-7
                                             (* y 1.05787412e-7)))))))))
            ans (* SQ2OPI (/ 1.0 (n/sqrt ax))
                   (- (* p1 (m/cos xx))
                      (* q1 (m/sin xx))))]
        (if (neg? x) (- ans) ans)))))

;; --- bessely0 ---

(deftm bessely0 [x :- Double] :- Double
  (if (<= x 8.0)
    (let [y (* x x)
          ans1 (+ -2957821389.0
                  (* y (+ 7062834065.0
                          (* y (+ -512359803.6
                                  (* y (+ 10879881.29
                                          (* y (+ -86327.92757
                                                  (* y 228.4622733))))))))))
          ans2 (+ 40076544269.0
                  (* y (+ 745249964.8
                          (* y (+ 7189466.438
                                  (* y (+ 47447.26470
                                          (* y (+ 226.1030244
                                                  y)))))))))
          j0 (besselj0 x)]
      (+ (/ ans1 ans2)
         (* TWO_OVER_PI j0 (m/log x))))
    (let [z (/ 8.0 x)
          y (* z z)
          xx (- x (* 0.25 n/pi))
          p0 (+ 1.0
                (* y (+ -1.098628627e-3
                        (* y (+ 2.734510407e-5
                                (* y (+ -2.073370639e-6
                                        (* y 2.093887211e-7))))))))
          q0 (* z
                (+ -0.01562499995
                   (* y (+ 1.430488765e-4
                           (* y (+ -6.911147651e-6
                                   (* y (+ 7.621095161e-7
                                           (* y -9.34945152e-8)))))))))]
      (* SQ2OPI (/ 1.0 (n/sqrt x))
         (+ (* p0 (m/sin xx))
            (* q0 (m/cos xx)))))))

;; --- bessely1 ---

(deftm bessely1 [x :- Double] :- Double
  (if (<= x 8.0)
    (let [y (* x x)
          ans1 (* x
                  (+ -4900604943000.0
                     (* y (+ 1275274390000.0
                             (* y (+ -51534866838.0
                                     (* y (+ 622785432.7
                                             (* y (+ -3130900.625
                                                     (* y 6516.196399)))))))))))
          ans2 (+ 24995805700000.0
                  (* y (+ 424441966400.0
                          (* y (+ 3733650367.0
                                  (* y (+ 22459040.02
                                          (* y (+ 103680.0
                                                  y)))))))))
          j1 (besselj1 x)]
      (+ (/ ans1 ans2)
         (* TWO_OVER_PI (- (* j1 (m/log x)) (/ 1.0 x)))))
    (let [z (/ 8.0 x)
          y (* z z)
          xx (- x (* 0.75 n/pi))
          p1 (+ 1.0
                (* y (+ 1.83105e-3
                        (* y (+ -3.516396496e-5
                                (* y (+ 2.457520174e-6
                                        (* y -2.40337019e-7))))))))
          q1 (* z
                (+ 0.04687499995
                   (* y (+ -2.002690873e-4
                           (* y (+ 8.449199096e-6
                                   (* y (+ -8.8228987e-7
                                           (* y 1.05787412e-7)))))))))]
      (* SQ2OPI (/ 1.0 (n/sqrt x))
         (+ (* p1 (m/sin xx))
            (* q1 (m/cos xx)))))))

;; --- besseli0 ---

(deftm besseli0 [x :- Double] :- Double
  (let [ax (n/abs x)]
    (if (<= ax 3.75)
      (let [t (/ ax 3.75)
            y (* t t)]
        (+ 1.0
           (* y (+ 3.5156229
                   (* y (+ 3.0899424
                           (* y (+ 1.2067492
                                   (* y (+ 0.2659732
                                           (* y (+ 0.0360768
                                                   (* y 0.0045813)))))))))))))
      (let [y (/ 3.75 ax)]
        (* (/ (m/exp ax) (n/sqrt ax))
           (+ 0.39894228
              (* y (+ 0.01328592
                      (* y (+ 0.00225319
                              (* y (+ -0.00157565
                                      (* y (+ 0.00916281
                                              (* y (+ -0.02057706
                                                      (* y (+ 0.02635537
                                                              (* y (+ -0.01647633
                                                                      (* y 0.00392377)))))))))))))))))))))

;; --- besseli1 ---

(deftm besseli1 [x :- Double] :- Double
  (let [ax (n/abs x)]
    (if (<= ax 3.75)
      (let [t (/ ax 3.75)
            y (* t t)
            ans (* ax
                   (+ 0.5
                      (* y (+ 0.87890594
                              (* y (+ 0.51498869
                                      (* y (+ 0.15084934
                                              (* y (+ 0.02658733
                                                      (* y (+ 0.00301532
                                                              (* y 0.00032411)))))))))))))]
        (if (neg? x) (- ans) ans))
      (let [y (/ 3.75 ax)
            ans (* (/ (m/exp ax) (n/sqrt ax))
                   (+ 0.39894228
                      (* y (+ -0.03988024
                              (* y (+ -0.00362018
                                      (* y (+ 0.00163801
                                              (* y (+ -0.01031555
                                                      (* y (+ 0.02282967
                                                              (* y (+ -0.02895312
                                                                      (* y (+ 0.01787654
                                                                              (* y -0.00420059)))))))))))))))))]
        (if (neg? x) (- ans) ans)))))

;; --- besselk0 ---

(deftm besselk0 [x :- Double] :- Double
  (if (<= x 2.0)
    (let [y (* 0.25 x x)
          i0 (besseli0 x)
          poly (+ 0.42278420
                  (* y (+ 0.23069756
                          (* y (+ 0.03488590
                                  (* y (+ 0.00262698
                                          (* y (+ 0.00010750
                                                  (* y 0.0000074))))))))))]
      (+ (- (* i0 (m/log (* 0.5 x))))
         -0.57721566
         (* y poly)))
    (let [y (/ 2.0 x)
          poly (+ 1.25331414
                  (* y (+ -0.07832358
                          (* y (+ 0.02189568
                                  (* y (+ -0.01062446
                                          (* y (+ 0.00587872
                                                  (* y (+ -0.00251540
                                                          (* y 0.00053208))))))))))))]
      (* (/ (m/exp (- x)) (n/sqrt x)) poly))))

;; --- besselk1 ---

(deftm besselk1 [x :- Double] :- Double
  (if (<= x 2.0)
    (let [y (* 0.25 x x)
          i1 (besseli1 x)
          poly (+ 0.15443144
                  (* y (+ -0.67278579
                          (* y (+ -0.18156897
                                  (* y (+ -0.01919402
                                          (* y (+ -0.00110404
                                                  (* y -0.00004686))))))))))]
      (+ (* i1 (m/log (* 0.5 x)))
         (/ 1.0 x)
         (* y poly)))
    (let [y (/ 2.0 x)
          poly (+ 1.25331414
                  (* y (+ 0.23498619
                          (* y (+ -0.03655620
                                  (* y (+ 0.01504268
                                          (* y (+ -0.00780353
                                                  (* y (+ 0.00325614
                                                          (* y -0.00068245))))))))))))]
      (* (/ (m/exp (- x)) (n/sqrt x)) poly))))

;; ================================================================
;; Other special functions
;; ================================================================

(deftm sinc (All [T] [x :- T] :- T
                 (if (< (n/abs x) 1e-20)
                   1.0
                   (let [pix (* n/pi x)]
                     (/ (m/sin pix) pix)))))

(deftm logit (All [T] [x :- T] :- T
                  (m/log (/ x (- 1.0 x)))))

(deftm expit (All [T] [x :- T] :- T
                  (/ 1.0 (+ 1.0 (m/exp (- x))))))

(deftm log1pexp (All [T] [x :- T] :- T
                     (cond
                       (< x -37.0) (m/exp x)
                       (< x 18.0)  (m/log1p (m/exp x))
                       (< x 33.6)  (+ x (m/exp (- x)))
                       :else        x)))

(deftm logaddexp (All [T] [a :- T, b :- T] :- T
                      (let [mx (n/max a b)]
                        (+ mx (m/log (+ (m/exp (- a mx)) (m/exp (- b mx))))))))

(deftm zeta [s :- Double] :- Double
  (if (<= s 1.0)
    (throw (ex-info "zeta(s) only implemented for s > 1" {:s s}))
    (let [n 12
          direct-sum (loop [k 1 acc 0.0]
                       (if (>= k n)
                         acc
                         (recur (inc k) (+ acc (n/pow (double k) (- s))))))
          ns (n/pow (double n) (- s))
          integral (/ (n/pow (double n) (- 1.0 s)) (- s 1.0))
          inv-n (/ 1.0 (double n))
          b2-term (* (/ 1.0 6.0) s ns inv-n)
          b4-term (* (/ -1.0 30.0)
                     (* s (+ s 1.0) (+ s 2.0))
                     ns (* inv-n inv-n inv-n)
                     (/ 1.0 24.0))
          b6-term (* (/ 1.0 42.0)
                     (* s (+ s 1.0) (+ s 2.0) (+ s 3.0) (+ s 4.0))
                     ns (* inv-n inv-n inv-n inv-n inv-n)
                     (/ 1.0 720.0))]
      (+ direct-sum integral (* 0.5 ns)
         b2-term b4-term b6-term))))
