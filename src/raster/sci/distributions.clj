(ns raster.sci.distributions
  "Probability distributions with typed multiple dispatch.

   Implements a Julia Distributions.jl-inspired hierarchy using
   defabstract for abstract categories and defvalue for concrete
   distributions. Dispatch on abstract types enables generic fallbacks
   (e.g., pdf from logpdf) while specialized methods provide performance.

   Usage:
     (require '[raster.sci.distributions :refer [->Normal logpdf pdf cdf sample mean variance]])
     (def d (->Normal 0.0 1.0))
     (logpdf d 0.0)    ;=> -0.9189...
     (mean d)           ;=> 0.0
     (sample d)         ;=> random draw"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue defabstract broadcast reduce!]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as mn]
            [raster.sci.special :as special])
  (:import [java.util.concurrent ThreadLocalRandom]))

;; ================================================================
;; Abstract hierarchy
;; ================================================================

(defabstract Distribution)
(defabstract Univariate :extends Distribution)
(defabstract Multivariate :extends Distribution)
(defabstract Continuous :extends Distribution)
(defabstract Discrete :extends Distribution)

;; ================================================================
;; Concrete distributions (dual inheritance via vector :implements)
;; ================================================================

(defvalue Normal [mu :- Double, sigma :- Double] :implements [Univariate Continuous])
(defvalue Uniform [a :- Double, b :- Double] :implements [Univariate Continuous])
(defvalue Exponential [lambda :- Double] :implements [Univariate Continuous])
(defvalue Gamma [alpha :- Double, beta :- Double] :implements [Univariate Continuous])
(defvalue Poisson [lambda :- Double] :implements [Univariate Discrete])

;; ================================================================
;; Special math helpers
;; ================================================================

(declare lgamma regularized-beta-cf)

(def ^:private ^:const LOG_2PI (m/log (* 2.0 mn/pi)))
(def ^:private ^:const SQRT_2 (mn/sqrt 2.0))
(def ^:private ^:const INV_SQRT_2 (/ 1.0 SQRT_2))
(def ^:private ^:const LOG_2 (m/log 2.0))

(deftm erf [x :- Double] :- Double
  (let [sign (if (neg? x) -1.0 1.0)
        x (mn/abs x)
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

(def ^:private ^:const INV_SQRT_PI (/ 2.0 (mn/sqrt mn/pi)))

(deftm erfinv [y :- Double] :- Double
  (if (== y 0.0)
    0.0
    ;; Initial approximation using Winitzki's formula
    (let [a  0.147
          ln1my2 (m/log (- 1.0 (* y y)))
          b  (+ (/ 2.0 (* mn/pi a)) (* 0.5 ln1my2))
          x0 (* (m/signum y)
                (mn/sqrt (- (mn/sqrt (- (* b b) (/ ln1my2 a))) b)))]
      ;; Newton-Raphson: x_{n+1} = x_n + (y - erf(x_n)) / (2/sqrt(pi) * exp(-x_n^2))
      (loop [x x0 iter 0]
        (if (>= iter 8)
          x
          (let [err (- y (erf x))
                deriv (* INV_SQRT_PI (m/exp (- (* x x))))
                x-new (+ x (/ err deriv))]
            (if (< (mn/abs err) 1e-15)
              x-new
              (recur x-new (inc iter)))))))))

(deftm lgamma [x :- Double] :- Double
  ;; For x >= 0.5, use Lanczos approximation (g=7, n=9)
  (if (< x 0.5)
    ;; Reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
    (- (m/log (/ mn/pi (m/sin (* mn/pi x))))
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
          ;; Compute series sum
          s (loop [i 7 acc 0.99999999999980993]
              (if (neg? i)
                acc
                (recur (dec i) (+ acc (/ (aget coeffs i) (+ x (double i) 1.0))))))]
      (+ (* 0.5 (m/log (* 2.0 mn/pi)))
         (* (+ x 0.5) (m/log t))
         (- t)
         (m/log s)))))

(deftm regularized-gamma-p [a :- Double, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (let [lga (lgamma a)]
      (loop [n 0 sum 0.0 term (/ 1.0 a)]
        (if (or (> n 200) (< (mn/abs term) (* 1e-14 (mn/abs sum))))
          (* sum (m/exp (- (* a (m/log x)) x lga)))
          (let [new-sum (+ sum term)
                new-term (* term (/ x (+ a (double n) 1.0)))]
            (recur (inc n) new-sum new-term)))))))

(deftm logsumexp [a :- Double, b :- Double] :- Double
  (let [mx (mn/max a b)]
    (+ mx (m/log (+ (m/exp (- a mx)) (m/exp (- b mx)))))))

;; ================================================================
;; logpdf / pdf — continuous distributions
;; ================================================================

(deftm logpdf [d :- Normal, x :- Double] :- Double
  (let [z (/ (- x (.mu d)) (.sigma d))]
    (* -0.5 (+ LOG_2PI (* 2.0 (m/log (.sigma d))) (* z z)))))

(deftm logpdf [d :- Uniform, x :- Double] :- Double
  (if (and (>= x (.a d)) (<= x (.b d)))
    (- (m/log (- (.b d) (.a d))))
    mn/neg-inf))

(deftm logpdf [d :- Exponential, x :- Double] :- Double
  (if (>= x 0.0)
    (- (m/log (.lambda d)) (* (.lambda d) x))
    mn/neg-inf))

(deftm logpdf [d :- Gamma, x :- Double] :- Double
  (if (> x 0.0)
    (let [a (.alpha d) b (.beta d)]
      (- (* (- a 1.0) (m/log x))
         (/ x b)
         (lgamma a)
         (* a (m/log b))))
    mn/neg-inf))

;; Generic logpdf fallbacks using polymorphic arithmetic (AD-compatible).
;; When x is a Dual number, these methods are selected over the Double-specific
;; ones, enabling automatic differentiation through logpdf.
(deftm logpdf [d :- Normal, x :- Object]
  (let [z (mn// (mn/- x (.mu d)) (.sigma d))]
    (mn/* -0.5 (mn/+ LOG_2PI (* 2.0 (m/log (.sigma d))) (mn/* z z)))))

(deftm logpdf [d :- Exponential, x :- Object]
  (mn/- (m/log (.lambda d)) (mn/* (.lambda d) x)))

;; Generic pdf from logpdf for any Continuous distribution
(deftm pdf [d :- Continuous, x :- Double] :- Double
  (m/exp (logpdf d x)))

;; ================================================================
;; logpmf / pmf — discrete distributions
;; ================================================================

(deftm logpmf [d :- Poisson, k :- Long] :- Double
  (if (>= k 0)
    (let [lam (.lambda d)]
      (- (* (double k) (m/log lam)) lam (lgamma (+ (double k) 1.0))))
    mn/neg-inf))

(deftm pmf [d :- Poisson, k :- Long] :- Double
  (m/exp (logpmf d k)))

;; ================================================================
;; CDF
;; ================================================================

(deftm cdf [d :- Normal, x :- Double] :- Double
  (let [z (/ (- x (.mu d)) (.sigma d))]
    (* 0.5 (+ 1.0 (erf (* INV_SQRT_2 z))))))

(deftm cdf [d :- Uniform, x :- Double] :- Double
  (cond
    (< x (.a d)) 0.0
    (> x (.b d)) 1.0
    :else (/ (- x (.a d)) (- (.b d) (.a d)))))

(deftm cdf [d :- Exponential, x :- Double] :- Double
  (if (>= x 0.0)
    (- 1.0 (m/exp (- (* (.lambda d) x))))
    0.0))

(deftm cdf [d :- Gamma, x :- Double] :- Double
  (if (> x 0.0)
    (regularized-gamma-p (.alpha d) (/ x (.beta d)))
    0.0))

(deftm cdf [d :- Poisson, x :- Double] :- Double
  (if (< x 0.0)
    0.0
    (let [k (long (m/floor x))
          lam (.lambda d)]
      ;; CDF = 1 - P(alpha, lambda) where alpha = k+1
      (- 1.0 (regularized-gamma-p (+ (double k) 1.0) lam)))))

;; ================================================================
;; Quantile (inverse CDF)
;; ================================================================

(deftm quantile [d :- Normal, p :- Double] :- Double
  (+ (.mu d) (* (.sigma d) SQRT_2 (erfinv (- (* 2.0 p) 1.0)))))

(deftm quantile [d :- Uniform, p :- Double] :- Double
  (+ (.a d) (* p (- (.b d) (.a d)))))

(deftm quantile [d :- Exponential, p :- Double] :- Double
  (/ (- (m/log (- 1.0 p))) (.lambda d)))

;; ================================================================
;; Statistics
;; ================================================================

;; --- Mean ---
(deftm mean [d :- Normal] :- Double (.mu d))
(deftm mean [d :- Uniform] :- Double (* 0.5 (+ (.a d) (.b d))))
(deftm mean [d :- Exponential] :- Double (/ 1.0 (.lambda d)))
(deftm mean [d :- Gamma] :- Double (* (.alpha d) (.beta d)))
(deftm mean [d :- Poisson] :- Double (.lambda d))

;; --- Variance ---
(deftm variance [d :- Normal] :- Double (* (.sigma d) (.sigma d)))
(deftm variance [d :- Uniform] :- Double
  (let [w (- (.b d) (.a d))]
    (/ (* w w) 12.0)))
(deftm variance [d :- Exponential] :- Double
  (/ 1.0 (* (.lambda d) (.lambda d))))
(deftm variance [d :- Gamma] :- Double
  (* (.alpha d) (.beta d) (.beta d)))
(deftm variance [d :- Poisson] :- Double (.lambda d))

;; --- Std (generic from variance) ---
(deftm std [d :- Distribution] :- Double
  (mn/sqrt (variance d)))

;; --- Entropy ---
(deftm entropy [d :- Normal] :- Double
  (* 0.5 (+ 1.0 LOG_2PI (* 2.0 (m/log (.sigma d))))))

(deftm entropy [d :- Uniform] :- Double
  (m/log (- (.b d) (.a d))))

(deftm entropy [d :- Exponential] :- Double
  (- 1.0 (m/log (.lambda d))))

;; --- Mode ---
(deftm mode [d :- Normal] :- Double (.mu d))
(deftm mode [d :- Uniform] :- Double (* 0.5 (+ (.a d) (.b d))))
(deftm mode [d :- Exponential] :- Double 0.0)
(deftm mode [d :- Gamma] :- Double
  (if (>= (.alpha d) 1.0)
    (* (- (.alpha d) 1.0) (.beta d))
    (throw (ex-info "Gamma mode undefined for alpha < 1" {:alpha (.alpha d)}))))
(deftm mode [d :- Poisson] :- Double (m/floor (.lambda d)))

;; --- Support bounds ---
(deftm support-min [d :- Normal] :- Double mn/neg-inf)
(deftm support-min [d :- Uniform] :- Double (.a d))
(deftm support-min [d :- Exponential] :- Double 0.0)
(deftm support-min [d :- Gamma] :- Double 0.0)
(deftm support-min [d :- Poisson] :- Double 0.0)

(deftm support-max [d :- Normal] :- Double mn/pos-inf)
(deftm support-max [d :- Uniform] :- Double (.b d))
(deftm support-max [d :- Exponential] :- Double mn/pos-inf)
(deftm support-max [d :- Gamma] :- Double mn/pos-inf)
(deftm support-max [d :- Poisson] :- Double mn/pos-inf)

;; ================================================================
;; Sampling
;; ================================================================

(deftm sample [d :- Normal] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (+ (.mu d) (* (.sigma d) (.nextGaussian rng)))))

(deftm sample [d :- Uniform] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (+ (.a d) (* (.nextDouble rng) (- (.b d) (.a d))))))

(deftm sample [d :- Exponential] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (/ (- (m/log (.nextDouble rng))) (.lambda d))))

(deftm sample [d :- Gamma] :- Double
  ;; Marsaglia and Tsang's method for alpha >= 1
  ;; For alpha < 1, use the transformation: X = Y^(1/alpha) where Y ~ Gamma(alpha+1)
  (let [rng (ThreadLocalRandom/current)
        a (.alpha d)
        b (.beta d)]
    (if (< a 1.0)
      ;; Ahrens-Dieter: Gamma(a) = Gamma(a+1) * U^(1/a)
      (let [g-a1 (let [d (- (+ a 1.0) (/ 1.0 3.0))
                       c (/ 1.0 (mn/sqrt (* 9.0 d)))]
                   (loop []
                     (let [x (.nextGaussian rng)
                           v (let [tmp (+ 1.0 (* c x))]
                               (* tmp tmp tmp))]
                       (if (and (> v 0.0)
                                (< (m/log (.nextDouble rng))
                                   (+ (* 0.5 x x) (* d (- 1.0 v (m/log v))))))
                         (* d v)
                         (recur)))))]
        (* b g-a1 (mn/pow (.nextDouble rng) (/ 1.0 a))))
      ;; Marsaglia-Tsang for alpha >= 1
      (let [d (- a (/ 1.0 3.0))
            c (/ 1.0 (mn/sqrt (* 9.0 d)))]
        (loop []
          (let [x (.nextGaussian rng)
                v (let [tmp (+ 1.0 (* c x))]
                    (* tmp tmp tmp))]
            (if (and (> v 0.0)
                     (< (m/log (.nextDouble rng))
                        (+ (* 0.5 x x) (* d (- 1.0 v (m/log v))))))
              (* b d v)
              (recur))))))))

(deftm sample [d :- Poisson] :- Double
  ;; Knuth's algorithm for small lambda; for large lambda use normal approx
  (let [lam (.lambda d)]
    (if (< lam 30.0)
      ;; Knuth's method
      (let [rng (ThreadLocalRandom/current)
            L (m/exp (- lam))]
        (loop [k 0 p 1.0]
          (if (<= p L)
            (double (dec k))
            (recur (inc k) (* p (.nextDouble rng))))))
      ;; Normal approximation for large lambda
      (let [rng (ThreadLocalRandom/current)]
        (m/round (+ lam (* (mn/sqrt lam) (.nextGaussian rng))))))))

(deftm sample-n [d :- Distribution, n :- Long]
  (let [out (double-array n)]
    (dotimes [i n]
      (aset out i (double (sample d))))
    out))

;; ================================================================
;; Fitting (MLE)
;; ================================================================

(deftm fit-normal [data :- (Array double)]
  (let [n (alength data)
        mu (/ (reduce! [acc 0.0] [data] (+ acc data))
              (double n))
        sigma (mn/sqrt (/ (reduce! [acc 0.0] [data]
                                   (let [d (- data mu)]
                                     (+ acc (* d d))))
                          (double n)))]
    (->Normal mu sigma)))

(deftm fit-exponential [data :- (Array double)]
  (let [n (alength data)
        sum (reduce! [acc 0.0] [data] (+ acc data))]
    (->Exponential (/ (double n) sum))))

;; ================================================================
;; Additional continuous distributions
;; ================================================================

(defvalue Beta [alpha :- Double, beta :- Double] :implements [Univariate Continuous])
(defvalue Cauchy [mu :- Double, sigma :- Double] :implements [Univariate Continuous])
(defvalue LogNormal [mu :- Double, sigma :- Double] :implements [Univariate Continuous])
(defvalue StudentT [nu :- Double] :implements [Univariate Continuous])
(defvalue Weibull [alpha :- Double, theta :- Double] :implements [Univariate Continuous])
(defvalue Rayleigh [sigma :- Double] :implements [Univariate Continuous])
(defvalue Pareto [alpha :- Double, theta :- Double] :implements [Univariate Continuous])
(defvalue Chisq [nu :- Double] :implements [Univariate Continuous])

;; Additional discrete distributions
(defvalue Bernoulli [p :- Double] :implements [Univariate Discrete])
(defvalue Binomial [n :- Long, p :- Double] :implements [Univariate Discrete])
(defvalue Geometric [p :- Double] :implements [Univariate Discrete])

;; ================================================================
;; Special math helpers for new distributions
;; ================================================================

(deftm lbeta [a :- Double, b :- Double] :- Double
  (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

(deftm regularized-beta-cf [x :- Double, a :- Double, b :- Double] :- Double
  (let [max-iter 200
        eps 3.0e-12
        fpmin 1.0e-30]
    (cond
      (<= x 0.0) 0.0
      (>= x 1.0) 1.0
      ;; Symmetry relation: when x > (a+1)/(a+b+2), use I_x(a,b) = 1 - I_{1-x}(b,a)
      (> x (/ (+ a 1.0) (+ a b 2.0)))
      (- 1.0 (regularized-beta-cf (- 1.0 x) b a))
      :else
      (let [;; Prefactor: x^a * (1-x)^b / (a * B(a,b))
            lnpfx (- (+ (* a (m/log x)) (* b (m/log (- 1.0 x))))
                     (lbeta a b) (m/log a))
            qab (+ a b)
            qap (+ a 1.0)
            qam (- a 1.0)
            ;; Initial: d = 1, c = 1, f = 1
            ;; First coefficient (m=0): a_1 = 1 (trivially)
            ;; Lentz's method modified start
            c0 1.0
            d0 (let [v (- 1.0 (/ (* qab x) qap))]
                 (if (< (mn/abs v) fpmin) (/ 1.0 fpmin) (/ 1.0 v)))
            h0 d0]
        (* (m/exp lnpfx)
           (loop [m 1, c c0, d d0, h h0]
             (if (>= m max-iter)
               h
               (let [dm (double m)
                     ;; Even term: a_{2m} = m*(b-m)*x / ((qam+2m)*(a+2m))
                     em (/ (* dm (- b dm) x)
                           (* (+ qam (* 2.0 dm)) (+ a (* 2.0 dm))))
                     d1 (+ 1.0 (* em d))
                     d1 (if (< (mn/abs d1) fpmin) fpmin d1)
                     d1 (/ 1.0 d1)
                     c1 (+ 1.0 (/ em c))
                     c1 (if (< (mn/abs c1) fpmin) fpmin c1)
                     h (* h d1 c1)
                     ;; Odd term: a_{2m+1} = -(a+m)*(qab+m)*x / ((a+2m)*(qap+2m))
                     ep (/ (- (* (+ a dm) (+ qab dm) x))
                           (* (+ a (* 2.0 dm)) (+ qap (* 2.0 dm))))
                     d2 (+ 1.0 (* ep d1))
                     d2 (if (< (mn/abs d2) fpmin) fpmin d2)
                     d2 (/ 1.0 d2)
                     c2 (+ 1.0 (/ ep c1))
                     c2 (if (< (mn/abs c2) fpmin) fpmin c2)
                     del (* d2 c2)
                     h (* h del)]
                 (if (< (mn/abs (- del 1.0)) eps)
                   h
                   (recur (inc m) c2 d2 h))))))))))

;; ================================================================
;; logpdf for new continuous distributions
;; ================================================================

(deftm logpdf [d :- Beta, x :- Double] :- Double
  (if (and (> x 0.0) (< x 1.0))
    (let [a (.alpha d) b (.beta d)]
      (+ (* (- a 1.0) (m/log x))
         (* (- b 1.0) (m/log (- 1.0 x)))
         (- (lbeta a b))))
    mn/neg-inf))

(deftm logpdf [d :- Cauchy, x :- Double] :- Double
  (let [z (/ (- x (.mu d)) (.sigma d))]
    (- (- (m/log mn/pi))
       (m/log (.sigma d))
       (m/log (+ 1.0 (* z z))))))

(deftm logpdf [d :- LogNormal, x :- Double] :- Double
  (if (> x 0.0)
    (let [lx (m/log x)
          z (/ (- lx (.mu d)) (.sigma d))]
      (- (- (* -0.5 (+ LOG_2PI (* z z)))
            (m/log (.sigma d)))
         lx))
    mn/neg-inf))

(deftm logpdf [d :- StudentT, x :- Double] :- Double
  (let [nu (.nu d)
        half-nu+1 (* 0.5 (+ nu 1.0))]
    (- (lgamma half-nu+1)
       (lgamma (* 0.5 nu))
       (* 0.5 (m/log (* nu mn/pi)))
       (* half-nu+1 (m/log (+ 1.0 (/ (* x x) nu)))))))

(deftm logpdf [d :- Weibull, x :- Double] :- Double
  (if (> x 0.0)
    (let [a (.alpha d) th (.theta d)
          z (/ x th)]
      (+ (m/log (/ a th))
         (* (- a 1.0) (m/log z))
         (- (mn/pow z a))))
    mn/neg-inf))

(deftm logpdf [d :- Rayleigh, x :- Double] :- Double
  (if (>= x 0.0)
    (let [s (.sigma d)
          s2 (* s s)]
      (- (m/log (/ x s2)) (/ (* x x) (* 2.0 s2))))
    mn/neg-inf))

(deftm logpdf [d :- Pareto, x :- Double] :- Double
  (if (>= x (.theta d))
    (let [a (.alpha d) th (.theta d)]
      (+ (m/log a)
         (* a (m/log th))
         (* (- (+ a 1.0)) (m/log x))))
    mn/neg-inf))

(deftm logpdf [d :- Chisq, x :- Double] :- Double
  (if (> x 0.0)
    (let [k (* 0.5 (.nu d))]
      (- (* (- k 1.0) (m/log x))
         (* 0.5 x)
         (* k LOG_2)
         (lgamma k)))
    mn/neg-inf))

;; ================================================================
;; logpmf for new discrete distributions
;; ================================================================

(deftm logpmf [d :- Bernoulli, k :- Long] :- Double
  (let [p (.p d)]
    (cond
      (== k 1) (m/log p)
      (== k 0) (m/log (- 1.0 p))
      :else mn/neg-inf)))

(deftm pmf [d :- Bernoulli, k :- Long] :- Double
  (m/exp (logpmf d k)))

(deftm logpmf [d :- Binomial, k :- Long] :- Double
  (let [n (.n d) p (.p d)]
    (if (and (>= k 0) (<= k n))
      (let [fk (double k) fn (double n)]
        (+ (- (lgamma (+ fn 1.0)) (lgamma (+ fk 1.0)) (lgamma (+ (- fn fk) 1.0)))
           (* fk (m/log p))
           (* (- fn fk) (m/log (- 1.0 p)))))
      mn/neg-inf)))

(deftm pmf [d :- Binomial, k :- Long] :- Double
  (m/exp (logpmf d k)))

(deftm logpmf [d :- Geometric, k :- Long] :- Double
  (if (>= k 0)
    (+ (m/log (.p d)) (* (double k) (m/log (- 1.0 (.p d)))))
    mn/neg-inf))

(deftm pmf [d :- Geometric, k :- Long] :- Double
  (m/exp (logpmf d k)))

;; ================================================================
;; CDF for new distributions
;; ================================================================

(deftm cdf [d :- Beta, x :- Double] :- Double
  (cond
    (<= x 0.0) 0.0
    (>= x 1.0) 1.0
    :else (regularized-beta-cf x (.alpha d) (.beta d))))

(deftm cdf [d :- Cauchy, x :- Double] :- Double
  (+ 0.5 (/ (m/atan (/ (- x (.mu d)) (.sigma d))) mn/pi)))

(deftm cdf [d :- LogNormal, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (let [z (/ (- (m/log x) (.mu d)) (.sigma d))]
      (* 0.5 (+ 1.0 (erf (* INV_SQRT_2 z)))))))

(deftm cdf [d :- StudentT, x :- Double] :- Double
  (let [nu (.nu d)
        t2 (* x x)
        xt (/ nu (+ nu t2))]
    (if (>= x 0.0)
      (- 1.0 (* 0.5 (regularized-beta-cf xt (* 0.5 nu) 0.5)))
      (* 0.5 (regularized-beta-cf xt (* 0.5 nu) 0.5)))))

(deftm cdf [d :- Weibull, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (- 1.0 (m/exp (- (mn/pow (/ x (.theta d)) (.alpha d)))))))

(deftm cdf [d :- Rayleigh, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (- 1.0 (m/exp (/ (- (* x x)) (* 2.0 (.sigma d) (.sigma d)))))))

(deftm cdf [d :- Pareto, x :- Double] :- Double
  (if (< x (.theta d))
    0.0
    (- 1.0 (mn/pow (/ (.theta d) x) (.alpha d)))))

(deftm cdf [d :- Chisq, x :- Double] :- Double
  (if (<= x 0.0)
    0.0
    (regularized-gamma-p (* 0.5 (.nu d)) (* 0.5 x))))

(deftm cdf [d :- Bernoulli, x :- Double] :- Double
  (cond
    (< x 0.0) 0.0
    (< x 1.0) (- 1.0 (.p d))
    :else 1.0))

(deftm cdf [d :- Binomial, x :- Double] :- Double
  (let [n (.n d) p (.p d)
        k (long (m/floor x))]
    (cond
      (< k 0) 0.0
      (>= k n) 1.0
      :else (- 1.0 (regularized-beta-cf p (+ (double k) 1.0) (double (- n k)))))))

(deftm cdf [d :- Geometric, x :- Double] :- Double
  (if (< x 0.0)
    0.0
    (- 1.0 (mn/pow (- 1.0 (.p d)) (+ (m/floor x) 1.0)))))

;; ================================================================
;; Mean for new distributions
;; ================================================================

(deftm mean [d :- Beta] :- Double
  (/ (.alpha d) (+ (.alpha d) (.beta d))))

(deftm mean [d :- Cauchy] :- Double
  Double/NaN)  ;; undefined

(deftm mean [d :- LogNormal] :- Double
  (m/exp (+ (.mu d) (* 0.5 (.sigma d) (.sigma d)))))

(deftm mean [d :- StudentT] :- Double
  (if (> (.nu d) 1.0) 0.0 Double/NaN))

(deftm mean [d :- Weibull] :- Double
  (* (.theta d) (m/exp (lgamma (+ 1.0 (/ 1.0 (.alpha d)))))))

(deftm mean [d :- Rayleigh] :- Double
  (* (.sigma d) (mn/sqrt (* 0.5 mn/pi))))

(deftm mean [d :- Pareto] :- Double
  (if (> (.alpha d) 1.0)
    (/ (* (.alpha d) (.theta d)) (- (.alpha d) 1.0))
    mn/pos-inf))

(deftm mean [d :- Chisq] :- Double
  (.nu d))

(deftm mean [d :- Bernoulli] :- Double
  (.p d))

(deftm mean [d :- Binomial] :- Double
  (* (double (.n d)) (.p d)))

(deftm mean [d :- Geometric] :- Double
  (/ (- 1.0 (.p d)) (.p d)))

;; ================================================================
;; Variance for new distributions
;; ================================================================

(deftm variance [d :- Beta] :- Double
  (let [a (.alpha d) b (.beta d)
        ab (+ a b)]
    (/ (* a b) (* ab ab (+ ab 1.0)))))

(deftm variance [d :- Cauchy] :- Double
  Double/NaN)  ;; undefined

(deftm variance [d :- LogNormal] :- Double
  (let [s2 (* (.sigma d) (.sigma d))]
    (* (- (m/exp s2) 1.0) (m/exp (+ (* 2.0 (.mu d)) s2)))))

(deftm variance [d :- StudentT] :- Double
  (let [nu (.nu d)]
    (cond
      (> nu 2.0) (/ nu (- nu 2.0))
      (> nu 1.0) mn/pos-inf
      :else Double/NaN)))

(deftm variance [d :- Weibull] :- Double
  (let [a (.alpha d) th (.theta d)
        g1 (m/exp (lgamma (+ 1.0 (/ 1.0 a))))
        g2 (m/exp (lgamma (+ 1.0 (/ 2.0 a))))]
    (* th th (- g2 (* g1 g1)))))

(deftm variance [d :- Rayleigh] :- Double
  (let [s (.sigma d)]
    (* s s (- 2.0 (* 0.5 mn/pi)))))

(deftm variance [d :- Pareto] :- Double
  (let [a (.alpha d) th (.theta d)]
    (if (> a 2.0)
      (/ (* th th a) (* (- a 1.0) (- a 1.0) (- a 2.0)))
      mn/pos-inf)))

(deftm variance [d :- Chisq] :- Double
  (* 2.0 (.nu d)))

(deftm variance [d :- Bernoulli] :- Double
  (* (.p d) (- 1.0 (.p d))))

(deftm variance [d :- Binomial] :- Double
  (* (double (.n d)) (.p d) (- 1.0 (.p d))))

(deftm variance [d :- Geometric] :- Double
  (/ (- 1.0 (.p d)) (* (.p d) (.p d))))

;; ================================================================
;; Sampling for new distributions
;; ================================================================

(deftm sample [d :- Beta] :- Double
  ;; Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
  (let [x (sample (->Gamma (.alpha d) 1.0))
        y (sample (->Gamma (.beta d) 1.0))]
    (/ x (+ x y))))

(deftm sample [d :- Cauchy] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (+ (.mu d) (* (.sigma d) (m/tan (* mn/pi (- (.nextDouble rng) 0.5)))))))

(deftm sample [d :- LogNormal] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (m/exp (+ (.mu d) (* (.sigma d) (.nextGaussian rng))))))

(deftm sample [d :- StudentT] :- Double
  ;; T = Z / sqrt(V/nu) where Z~N(0,1), V~Chisq(nu)
  (let [rng (ThreadLocalRandom/current)
        z (.nextGaussian rng)
        v (sample (->Chisq (.nu d)))]
    (/ z (mn/sqrt (/ v (.nu d))))))

(deftm sample [d :- Weibull] :- Double
  (let [rng (ThreadLocalRandom/current)
        u (.nextDouble rng)]
    (* (.theta d) (mn/pow (- (m/log (- 1.0 u))) (/ 1.0 (.alpha d))))))

(deftm sample [d :- Rayleigh] :- Double
  (let [rng (ThreadLocalRandom/current)
        u (.nextDouble rng)]
    (* (.sigma d) (mn/sqrt (* -2.0 (m/log (- 1.0 u)))))))

(deftm sample [d :- Pareto] :- Double
  (let [rng (ThreadLocalRandom/current)
        u (.nextDouble rng)]
    (/ (.theta d) (mn/pow u (/ 1.0 (.alpha d))))))

(deftm sample [d :- Chisq] :- Double
  ;; Chisq(nu) = Gamma(nu/2, 2)
  (sample (->Gamma (* 0.5 (.nu d)) 2.0)))

(deftm sample [d :- Bernoulli] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (if (< (.nextDouble rng) (.p d)) 1.0 0.0)))

(deftm sample [d :- Binomial] :- Double
  ;; Direct method: sum of Bernoulli trials
  (let [rng (ThreadLocalRandom/current)
        n (.n d) p (.p d)]
    (loop [i 0 count 0.0]
      (if (>= i n)
        count
        (recur (inc i) (if (< (.nextDouble rng) p) (+ count 1.0) count))))))

(deftm sample [d :- Geometric] :- Double
  (let [rng (ThreadLocalRandom/current)]
    (m/floor (/ (m/log (.nextDouble rng)) (m/log (- 1.0 (.p d)))))))

;; ================================================================
;; Additional distributions (Phase 2)
;; ================================================================

(defvalue Laplace [mu :- Double, b :- Double] :implements [Univariate Continuous])
(defvalue Gumbel [mu :- Double, beta :- Double] :implements [Univariate Continuous])
(defvalue InverseGamma [alpha :- Double, beta :- Double] :implements [Univariate Continuous])
(defvalue NegativeBinomial [r :- Double, p :- Double] :implements [Univariate Discrete])
(defvalue Hypergeometric [N :- Long, K :- Long, nn :- Long] :implements [Univariate Discrete])
(defvalue Dirichlet [alpha :- (Array double), k :- Long] :implements [Multivariate Continuous])
(defvalue MultivariateNormal [mu :- (Array double), sigma :- (Array double), dim :- Long] :implements [Multivariate Continuous])

;; --- Laplace ---

(deftm logpdf [d :- Laplace, x :- Double] :- Double
  (- (- (m/log (* 2.0 (.b d))))
     (/ (mn/abs (- x (.mu d))) (.b d))))

(deftm cdf [d :- Laplace, x :- Double] :- Double
  (let [z (/ (- x (.mu d)) (.b d))]
    (if (<= x (.mu d))
      (* 0.5 (m/exp z))
      (- 1.0 (* 0.5 (m/exp (- z)))))))

(deftm mean [d :- Laplace] :- Double (.mu d))
(deftm variance [d :- Laplace] :- Double (* 2.0 (.b d) (.b d)))

(deftm sample [d :- Laplace] :- Double
  (let [rng (ThreadLocalRandom/current)
        u (- (.nextDouble rng) 0.5)]
    (- (.mu d) (* (.b d) (m/signum u) (m/log (- 1.0 (* 2.0 (mn/abs u))))))))

;; --- Gumbel ---

(deftm logpdf [d :- Gumbel, x :- Double] :- Double
  (let [z (/ (- x (.mu d)) (.beta d))]
    (- (- z) (m/exp (- z)) (m/log (.beta d)))))

(deftm cdf [d :- Gumbel, x :- Double] :- Double
  (m/exp (- (m/exp (- (/ (- x (.mu d)) (.beta d)))))))

(deftm mean [d :- Gumbel] :- Double
  (+ (.mu d) (* (.beta d) 0.5772156649015329))) ;; Euler-Mascheroni

(deftm variance [d :- Gumbel] :- Double
  (* (.beta d) (.beta d) (/ (* mn/pi mn/pi) 6.0)))

(deftm sample [d :- Gumbel] :- Double
  (let [rng (ThreadLocalRandom/current)
        u (.nextDouble rng)]
    (- (.mu d) (* (.beta d) (m/log (- (m/log u)))))))

;; --- InverseGamma ---

(deftm logpdf [d :- InverseGamma, x :- Double] :- Double
  (if (> x 0.0)
    (let [a (.alpha d) b (.beta d)]
      (- (* a (m/log b))
         (lgamma a)
         (* (+ a 1.0) (m/log x))
         (/ b x)))
    mn/neg-inf))

(deftm cdf [d :- InverseGamma, x :- Double] :- Double
  (if (<= x 0.0) 0.0
      (- 1.0 (special/gammainc (.alpha d) (/ (.beta d) x)))))

(deftm mean [d :- InverseGamma] :- Double
  (if (> (.alpha d) 1.0)
    (/ (.beta d) (- (.alpha d) 1.0))
    mn/pos-inf))

(deftm variance [d :- InverseGamma] :- Double
  (let [a (.alpha d) b (.beta d)]
    (if (> a 2.0)
      (/ (* b b) (* (- a 1.0) (- a 1.0) (- a 2.0)))
      mn/pos-inf)))

(deftm sample [d :- InverseGamma] :- Double
  ;; InverseGamma(a,b) = 1/Gamma(a,1/b)
  (/ 1.0 (sample (->Gamma (.alpha d) (/ 1.0 (.beta d))))))

;; --- NegativeBinomial ---

(deftm logpmf [d :- NegativeBinomial, k :- Long] :- Double
  (if (>= k 0)
    (let [r (.r d) p (.p d) fk (double k)]
      (+ (- (lgamma (+ r fk)) (lgamma (+ fk 1.0)) (lgamma r))
         (* r (m/log p))
         (* fk (m/log (- 1.0 p)))))
    mn/neg-inf))

(deftm pmf [d :- NegativeBinomial, k :- Long] :- Double
  (m/exp (logpmf d k)))

(deftm mean [d :- NegativeBinomial] :- Double
  (/ (* (.r d) (- 1.0 (.p d))) (.p d)))

(deftm variance [d :- NegativeBinomial] :- Double
  (/ (* (.r d) (- 1.0 (.p d))) (* (.p d) (.p d))))

(deftm sample [d :- NegativeBinomial] :- Double
  ;; NB(r,p) = Poisson(Gamma(r, (1-p)/p))
  (let [g (sample (->Gamma (.r d) (/ (- 1.0 (.p d)) (.p d))))]
    (sample (->Poisson g))))

;; --- Hypergeometric ---

(deftm logpmf [d :- Hypergeometric, k :- Long] :- Double
  (let [N (double (.N d))
        K (double (.K d))
        n (double (.nn d))
        fk (double k)]
    (if (and (>= k (long (mn/max 0.0 (- n (- N K)))))
             (<= k (long (mn/min n K))))
      (+ (- (+ (lgamma (+ K 1.0)) (lgamma (+ (- N K) 1.0))
               (lgamma (+ n 1.0)) (lgamma (+ (- N n) 1.0)))
            (+ (lgamma (+ N 1.0))
               (lgamma (+ fk 1.0)) (lgamma (+ (- K fk) 1.0))
               (lgamma (+ (- n fk) 1.0)) (lgamma (+ (- N K n) fk 1.0)))))
      mn/neg-inf)))

(deftm pmf [d :- Hypergeometric, k :- Long] :- Double
  (m/exp (logpmf d k)))

(deftm mean [d :- Hypergeometric] :- Double
  (/ (* (double (.nn d)) (double (.K d))) (double (.N d))))

(deftm variance [d :- Hypergeometric] :- Double
  (let [N (double (.N d)) K (double (.K d)) n (double (.nn d))]
    (/ (* n K (- N K) (- N n))
       (* N N (- N 1.0)))))

;; ================================================================
;; Dirichlet distribution
;; ================================================================

(deftm logpdf [d :- Dirichlet, x :- (Array double)] :- Double
  ;; logpdf = sum((alpha_i - 1) * log(x_i)) - lbeta(alpha)
  ;; lbeta(alpha) = sum(lgamma(alpha_i)) - lgamma(sum(alpha_i))
  (let [alpha (.alpha d)
        k (.k d)
        lbeta-val (loop [i 0 lg-sum 0.0 a-sum 0.0]
                    (if (>= i k)
                      (- lg-sum (lgamma a-sum))
                      (let [ai (aget alpha i)]
                        (recur (inc i)
                               (+ lg-sum (lgamma ai))
                               (+ a-sum ai)))))]
    (loop [i 0 acc 0.0]
      (if (>= i k)
        (- acc lbeta-val)
        (recur (inc i)
               (+ acc (* (- (aget alpha i) 1.0) (m/log (aget x i)))))))))

(deftm sample [d :- Dirichlet] :- (Array double)
  ;; Sample from Dirichlet by sampling Gamma(alpha_i, 1) and normalizing
  (let [alpha (.alpha d)
        k (.k d)
        result (double-array k)
        rng (ThreadLocalRandom/current)]
    (loop [i 0 total 0.0]
      (if (>= i k)
        (do (dotimes [j k]
              (aset result j (/ (aget result j) total)))
            result)
        (let [;; Sample Gamma(alpha_i, 1) via Marsaglia-Tsang
              ai (aget alpha i)
              g (sample (->Gamma ai 1.0))
              _ (aset result i g)]
          (recur (inc i) (+ total g)))))))

(deftm mean [d :- Dirichlet] :- (Array double)
  (let [alpha (.alpha d)
        k (.k d)
        a-sum (reduce! [acc 0.0] [alpha] (+ acc alpha))]
    (broadcast [alpha] (/ alpha a-sum))))

;; ================================================================
;; Multivariate Normal distribution
;; ================================================================
;; sigma is the FULL covariance matrix stored as flat double[] of size dim*dim

(deftm logpdf [d :- MultivariateNormal, x :- (Array double)] :- Double
  ;; logpdf = -0.5 * (k*ln(2pi) + ln|Sigma| + (x-mu)^T Sigma^{-1} (x-mu))
  ;; For simplicity, use Cholesky decomposition of Sigma
  (let [mu (.mu d)
        sigma (.sigma d)
        dim (.dim d)
        ;; Compute (x - mu)
        diff (double-array dim)
        _ (dotimes [i dim]
            (aset diff i (- (aget x i) (aget mu i))))
        ;; Cholesky of sigma: L such that L*L^T = Sigma
        ;; Inline Cholesky to avoid circular dep on linalg
        L (double-array (* dim dim))
        _ (acopy! sigma 0 L 0 (* dim dim))
        _ (loop [j 0]
            (when (< j dim)
              ;; Diagonal element
              (let [sum-sq (loop [k 0 acc 0.0]
                             (if (>= k j) acc
                                 (let [ljk (aget L (+ (* j dim) k))]
                                   (recur (inc k) (+ acc (* ljk ljk))))))
                    ljj (mn/sqrt (- (aget L (+ (* j dim) j)) sum-sq))]
                (aset L (+ (* j dim) j) ljj)
                ;; Off-diagonal elements in column j
                (loop [i (inc j)]
                  (when (< i dim)
                    (let [sum-prod (loop [k 0 acc 0.0]
                                     (if (>= k j) acc
                                         (recur (inc k)
                                                (+ acc (* (aget L (+ (* i dim) k))
                                                          (aget L (+ (* j dim) k)))))))
                          lij (/ (- (aget L (+ (* i dim) j)) sum-prod) ljj)]
                      (aset L (+ (* i dim) j) lij)
                      ;; Zero upper triangle
                      (aset L (+ (* j dim) i) 0.0))
                    (recur (inc i)))))
              (recur (inc j))))
        ;; log|Sigma| = 2 * sum(log(L_ii))
        log-det (loop [i 0 acc 0.0]
                  (if (>= i dim) (* 2.0 acc)
                      (recur (inc i) (+ acc (m/log (aget L (+ (* i dim) i)))))))
        ;; Solve L*y = diff (forward substitution)
        y (double-array dim)
        _ (loop [i 0]
            (when (< i dim)
              (let [s (loop [j 0 acc 0.0]
                        (if (>= j i) acc
                            (recur (inc j)
                                   (+ acc (* (aget L (+ (* i dim) j))
                                             (aget y j))))))]
                (aset y i (/ (- (aget diff i) s)
                             (aget L (+ (* i dim) i)))))
              (recur (inc i))))
        ;; Mahalanobis distance = y^T * y
        mahal (loop [i 0 acc 0.0]
                (if (>= i dim) acc
                    (recur (inc i) (+ acc (* (aget y i) (aget y i))))))]
    (* -0.5 (+ (* (double dim) (m/log (* 2.0 mn/pi)))
               log-det
               mahal))))

(deftm sample [d :- MultivariateNormal] :- (Array double)
  ;; Sample via Cholesky: z ~ N(0,I), x = mu + L*z
  (let [mu (.mu d)
        sigma (.sigma d)
        dim (.dim d)
        rng (ThreadLocalRandom/current)
        ;; Cholesky
        L (double-array (* dim dim))
        _ (acopy! sigma 0 L 0 (* dim dim))
        _ (loop [j 0]
            (when (< j dim)
              (let [sum-sq (loop [k 0 acc 0.0]
                             (if (>= k j) acc
                                 (let [ljk (aget L (+ (* j dim) k))]
                                   (recur (inc k) (+ acc (* ljk ljk))))))
                    ljj (mn/sqrt (- (aget L (+ (* j dim) j)) sum-sq))]
                (aset L (+ (* j dim) j) ljj)
                (loop [i (inc j)]
                  (when (< i dim)
                    (let [sum-prod (loop [k 0 acc 0.0]
                                     (if (>= k j) acc
                                         (recur (inc k)
                                                (+ acc (* (aget L (+ (* i dim) k))
                                                          (aget L (+ (* j dim) k)))))))
                          lij (/ (- (aget L (+ (* i dim) j)) sum-prod) ljj)]
                      (aset L (+ (* i dim) j) lij)
                      (aset L (+ (* j dim) i) 0.0))
                    (recur (inc i)))))
              (recur (inc j))))
        ;; Standard normal samples
        z (double-array dim)
        _ (dotimes [i dim]
            (aset z i (.nextGaussian rng)))
        ;; x = mu + L*z
        result (double-array dim)]
    (dotimes [i dim]
      (let [s (loop [j 0 acc 0.0]
                (if (> j i) acc
                    (recur (inc j)
                           (+ acc (* (aget L (+ (* i dim) j))
                                     (aget z j))))))]
        (aset result i (+ (aget mu i) s))))
    result))

(deftm mean [d :- MultivariateNormal] :- (Array double)
  (let [result (double-array (.dim d))]
    (acopy! (.mu d) 0 result 0 (.dim d))
    result))
