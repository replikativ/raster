(ns raster.sci.distributions-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.distributions :refer [->Normal ->Uniform ->Exponential ->Gamma ->Poisson
                                              ->Beta ->Cauchy ->LogNormal ->StudentT
                                              ->Weibull ->Rayleigh ->Pareto ->Chisq
                                              ->Bernoulli ->Binomial ->Geometric
                                              logpdf pdf cdf quantile
                                              logpmf pmf
                                              mean variance std entropy mode
                                              support-min support-max
                                              sample sample-n
                                              fit-normal fit-exponential]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Normal distribution
;; ================================================================

(deftest normal-logpdf-test
  (testing "standard normal at 0"
    ;; logpdf(0) = -0.5 * log(2*pi) ≈ -0.91893853
    (is (approx= -0.9189385332046727 (logpdf (->Normal 0.0 1.0) 0.0))))
  (testing "standard normal at 1"
    (is (approx= -1.4189385332046727 (logpdf (->Normal 0.0 1.0) 1.0))))
  (testing "pdf = exp(logpdf)"
    (let [d (->Normal 2.0 3.0) x 1.5]
      (is (approx= (Math/exp (logpdf d x)) (pdf d x))))))

(deftest normal-cdf-test
  (testing "standard normal CDF at 0"
    (is (approx= 0.5 (cdf (->Normal 0.0 1.0) 0.0))))
  (testing "standard normal CDF at +inf"
    (is (approx= 1.0 (cdf (->Normal 0.0 1.0) 10.0) 1e-4)))
  (testing "standard normal CDF at -inf"
    (is (approx= 0.0 (cdf (->Normal 0.0 1.0) -10.0) 1e-4))))

(deftest normal-quantile-test
  (testing "quantile 0.5 = mean"
    (is (approx= 0.0 (quantile (->Normal 0.0 1.0) 0.5) 1e-4)))
  (testing "quantile roundtrip"
    (let [d (->Normal 2.0 3.0)
          p 0.75]
      (is (approx= p (cdf d (quantile d p)) 1e-4)))))

(deftest normal-stats-test
  (testing "mean"
    (is (approx= 2.0 (mean (->Normal 2.0 3.0)))))
  (testing "variance"
    (is (approx= 9.0 (variance (->Normal 2.0 3.0)))))
  (testing "std"
    (is (approx= 3.0 (std (->Normal 2.0 3.0)))))
  (testing "entropy"
    ;; entropy = 0.5 * (1 + log(2*pi*sigma^2))
    (let [sigma 1.0
          expected (* 0.5 (+ 1.0 (Math/log (* 2.0 Math/PI sigma sigma))))]
      (is (approx= expected (entropy (->Normal 0.0 sigma))))))
  (testing "mode"
    (is (approx= 5.0 (mode (->Normal 5.0 2.0))))))

;; ================================================================
;; Uniform distribution
;; ================================================================

(deftest uniform-test
  (testing "logpdf inside support"
    (is (approx= (- (Math/log 2.0)) (logpdf (->Uniform 0.0 2.0) 1.0))))
  (testing "logpdf outside support"
    (is (Double/isInfinite (logpdf (->Uniform 0.0 1.0) 2.0))))
  (testing "cdf"
    (is (approx= 0.5 (cdf (->Uniform 0.0 1.0) 0.5)))
    (is (approx= 0.0 (cdf (->Uniform 0.0 1.0) -1.0)))
    (is (approx= 1.0 (cdf (->Uniform 0.0 1.0) 2.0))))
  (testing "quantile"
    (is (approx= 0.25 (quantile (->Uniform 0.0 1.0) 0.25))))
  (testing "mean"
    (is (approx= 0.5 (mean (->Uniform 0.0 1.0)))))
  (testing "variance"
    (is (approx= (/ 1.0 12.0) (variance (->Uniform 0.0 1.0))))))

;; ================================================================
;; Exponential distribution
;; ================================================================

(deftest exponential-test
  (testing "logpdf at 0"
    ;; logpdf(0; lambda=2) = log(2) - 0 = log(2)
    (is (approx= (Math/log 2.0) (logpdf (->Exponential 2.0) 0.0))))
  (testing "logpdf negative"
    (is (Double/isInfinite (logpdf (->Exponential 1.0) -1.0))))
  (testing "cdf"
    (is (approx= 0.0 (cdf (->Exponential 1.0) 0.0)))
    (is (approx= (- 1.0 (Math/exp -1.0)) (cdf (->Exponential 1.0) 1.0))))
  (testing "quantile roundtrip"
    (let [d (->Exponential 2.0) p 0.9]
      (is (approx= p (cdf d (quantile d p)) 1e-4))))
  (testing "mean"
    (is (approx= 0.5 (mean (->Exponential 2.0)))))
  (testing "variance"
    (is (approx= 0.25 (variance (->Exponential 2.0))))))

;; ================================================================
;; Gamma distribution
;; ================================================================

(deftest gamma-test
  (testing "Gamma(1,1) = Exponential(1)"
    ;; logpdf(1; Gamma(1,1)) should equal logpdf(1; Exp(1)) = -1
    (is (approx= -1.0 (logpdf (->Gamma 1.0 1.0) 1.0) 1e-4)))
  (testing "logpdf negative x"
    (is (Double/isInfinite (logpdf (->Gamma 2.0 1.0) -1.0))))
  (testing "mean"
    (is (approx= 6.0 (mean (->Gamma 2.0 3.0)))))
  (testing "variance"
    (is (approx= 18.0 (variance (->Gamma 2.0 3.0)))))
  (testing "cdf at 0"
    (is (approx= 0.0 (cdf (->Gamma 2.0 1.0) 0.0)))))

;; ================================================================
;; Poisson distribution
;; ================================================================

(deftest poisson-test
  (testing "pmf at mode"
    ;; P(X=5; lambda=5) ≈ 0.17547
    (is (approx= 0.175467 (pmf (->Poisson 5.0) 5) 1e-4)))
  (testing "logpmf at 0"
    ;; log(exp(-lambda)) = -lambda
    (is (approx= -5.0 (logpmf (->Poisson 5.0) 0) 1e-4)))
  (testing "mean"
    (is (approx= 3.0 (mean (->Poisson 3.0)))))
  (testing "variance"
    (is (approx= 3.0 (variance (->Poisson 3.0)))))
  (testing "cdf"
    ;; CDF(0; lambda=1) = P(X<=0) = exp(-1) ≈ 0.3679
    (is (approx= (Math/exp -1.0) (cdf (->Poisson 1.0) 0.0) 1e-3))))

;; ================================================================
;; Sampling smoke tests
;; ================================================================

(deftest sampling-test
  (testing "normal sample returns finite double"
    (let [d (->Normal 0.0 1.0)
          s (sample d)]
      (is (Double/isFinite s))))
  (testing "sample-n returns correct length"
    (let [d (->Normal 0.0 1.0)
          arr (sample-n d 1000)]
      (is (= 1000 (alength arr)))))
  (testing "uniform samples within bounds"
    (let [d (->Uniform 2.0 5.0)
          arr (sample-n d 100)]
      (is (every? #(and (>= % 2.0) (<= % 5.0)) arr))))
  (testing "exponential samples non-negative"
    (let [arr (sample-n (->Exponential 1.0) 100)]
      (is (every? #(>= % 0.0) arr))))
  (testing "poisson samples non-negative"
    (let [arr (sample-n (->Poisson 3.0) 100)]
      (is (every? #(>= % 0.0) arr)))))

;; ================================================================
;; Fitting
;; ================================================================

(deftest fit-test
  (testing "fit-normal recovers parameters"
    (let [d (->Normal 5.0 2.0)
          data (sample-n d 10000)
          fitted (fit-normal data)]
      (is (approx= 5.0 (mean fitted) 0.2))
      (is (approx= 2.0 (.sigma fitted) 0.2))))
  (testing "fit-exponential recovers rate"
    (let [d (->Exponential 3.0)
          data (sample-n d 10000)
          fitted (fit-exponential data)]
      (is (approx= 3.0 (.lambda fitted) 0.3)))))

;; ================================================================
;; Support bounds
;; ================================================================

(deftest support-test
  (testing "normal support"
    (is (= Double/NEGATIVE_INFINITY (support-min (->Normal 0.0 1.0))))
    (is (= Double/POSITIVE_INFINITY (support-max (->Normal 0.0 1.0)))))
  (testing "uniform support"
    (is (approx= 2.0 (support-min (->Uniform 2.0 5.0))))
    (is (approx= 5.0 (support-max (->Uniform 2.0 5.0))))))

;; ================================================================
;; Beta distribution
;; ================================================================

(deftest beta-test
  (testing "Beta(1,1) = Uniform(0,1)"
    ;; logpdf should be 0 everywhere on (0,1), i.e. pdf = 1
    (is (approx= 0.0 (logpdf (->Beta 1.0 1.0) 0.5) 1e-4)))
  (testing "Beta(2,2) symmetric, peak at 0.5"
    (let [d (->Beta 2.0 2.0)]
      ;; pdf(0.5) = 6 * 0.5 * 0.5 = 1.5
      (is (approx= 1.5 (pdf d 0.5) 1e-3))))
  (testing "mean"
    (is (approx= 0.4 (mean (->Beta 2.0 3.0)))))
  (testing "variance"
    ;; Var = ab / ((a+b)^2 * (a+b+1))
    (is (approx= (/ (* 2.0 3.0) (* 25.0 6.0)) (variance (->Beta 2.0 3.0)) 1e-4)))
  (testing "cdf at 0.5 for Beta(1,1)"
    (is (approx= 0.5 (cdf (->Beta 1.0 1.0) 0.5) 1e-3)))
  (testing "sampling"
    (let [arr (sample-n (->Beta 2.0 5.0) 100)]
      (is (every? #(and (> % 0.0) (< % 1.0)) arr)))))

;; ================================================================
;; Cauchy distribution
;; ================================================================

(deftest cauchy-test
  (testing "logpdf at center"
    ;; logpdf(0; 0, 1) = -log(pi) - log(1 + 0) = -log(pi)
    (is (approx= (- (Math/log Math/PI)) (logpdf (->Cauchy 0.0 1.0) 0.0))))
  (testing "cdf at center"
    (is (approx= 0.5 (cdf (->Cauchy 0.0 1.0) 0.0))))
  (testing "mean is NaN"
    (is (Double/isNaN (mean (->Cauchy 0.0 1.0)))))
  (testing "variance is NaN"
    (is (Double/isNaN (variance (->Cauchy 0.0 1.0))))))

;; ================================================================
;; LogNormal distribution
;; ================================================================

(deftest lognormal-test
  (testing "logpdf at exp(mu)"
    (let [d (->LogNormal 0.0 1.0)]
      ;; At x=1 (=exp(0)): logpdf = -0.5*log(2*pi) - log(1) - 0.5*0 = -0.5*log(2*pi)
      (is (approx= (* -0.5 (Math/log (* 2.0 Math/PI))) (logpdf d 1.0) 1e-4))))
  (testing "cdf"
    (is (approx= 0.5 (cdf (->LogNormal 0.0 1.0) 1.0) 1e-3)))
  (testing "mean"
    ;; E[X] = exp(mu + sigma^2/2)
    (is (approx= (Math/exp 0.5) (mean (->LogNormal 0.0 1.0)) 1e-4)))
  (testing "sampling positive"
    (let [arr (sample-n (->LogNormal 0.0 1.0) 100)]
      (is (every? pos? arr)))))

;; ================================================================
;; Student's t distribution
;; ================================================================

(deftest student-t-test
  (testing "logpdf at 0 for nu=1 (Cauchy)"
    ;; t(0; nu=1) = 1/(pi*(1+0)) = 1/pi → logpdf ≈ -log(pi)
    (is (approx= (- (Math/log Math/PI)) (logpdf (->StudentT 1.0) 0.0) 1e-3)))
  (testing "mean for nu>1"
    (is (approx= 0.0 (mean (->StudentT 5.0)))))
  (testing "variance for nu>2"
    ;; Var = nu/(nu-2) = 5/3
    (is (approx= (/ 5.0 3.0) (variance (->StudentT 5.0)) 1e-4)))
  (testing "cdf at 0 is 0.5"
    (is (approx= 0.5 (cdf (->StudentT 3.0) 0.0) 1e-3))))

;; ================================================================
;; Weibull distribution
;; ================================================================

(deftest weibull-test
  (testing "Weibull(1,1) = Exponential(1)"
    (is (approx= (logpdf (->Exponential 1.0) 1.0)
                 (logpdf (->Weibull 1.0 1.0) 1.0) 1e-4)))
  (testing "cdf"
    ;; CDF(x; 1, 1) = 1 - exp(-x)
    (is (approx= (- 1.0 (Math/exp -1.0)) (cdf (->Weibull 1.0 1.0) 1.0) 1e-4)))
  (testing "sampling positive"
    (let [arr (sample-n (->Weibull 2.0 1.0) 100)]
      (is (every? pos? arr)))))

;; ================================================================
;; Rayleigh distribution
;; ================================================================

(deftest rayleigh-test
  (testing "mean"
    ;; E[X] = sigma * sqrt(pi/2)
    (is (approx= (* 1.0 (Math/sqrt (* 0.5 Math/PI))) (mean (->Rayleigh 1.0)) 1e-4)))
  (testing "cdf"
    ;; CDF(x; sigma) = 1 - exp(-x^2/(2*sigma^2))
    (is (approx= (- 1.0 (Math/exp -0.5)) (cdf (->Rayleigh 1.0) 1.0) 1e-4)))
  (testing "sampling non-negative"
    (let [arr (sample-n (->Rayleigh 2.0) 100)]
      (is (every? #(>= % 0.0) arr)))))

;; ================================================================
;; Pareto distribution
;; ================================================================

(deftest pareto-test
  (testing "logpdf at threshold"
    ;; logpdf(1; alpha=2, theta=1) = log(2) + 2*log(1) - 3*log(1) = log(2)
    (is (approx= (Math/log 2.0) (logpdf (->Pareto 2.0 1.0) 1.0) 1e-4)))
  (testing "logpdf below threshold"
    (is (= Double/NEGATIVE_INFINITY (logpdf (->Pareto 2.0 1.0) 0.5))))
  (testing "mean"
    ;; E[X] = alpha*theta/(alpha-1) = 2*1/1 = 2
    (is (approx= 2.0 (mean (->Pareto 2.0 1.0)) 1e-4)))
  (testing "cdf"
    ;; CDF(2; alpha=2, theta=1) = 1 - (1/2)^2 = 0.75
    (is (approx= 0.75 (cdf (->Pareto 2.0 1.0) 2.0) 1e-4))))

;; ================================================================
;; Chi-squared distribution
;; ================================================================

(deftest chisq-test
  (testing "Chisq(2) = Exponential(0.5)"
    ;; Chisq(2) = Gamma(1, 2). At x=1: logpdf should match
    (let [chisq-lp (logpdf (->Chisq 2.0) 1.0)
          ;; Exponential(rate=0.5): logpdf = log(0.5) - 0.5*1 = -log(2) - 0.5
          exp-lp (logpdf (->Exponential 0.5) 1.0)]
      (is (approx= exp-lp chisq-lp 1e-3))))
  (testing "mean"
    (is (approx= 5.0 (mean (->Chisq 5.0)))))
  (testing "variance"
    (is (approx= 10.0 (variance (->Chisq 5.0))))))

;; ================================================================
;; Bernoulli distribution
;; ================================================================

(deftest bernoulli-test
  (testing "pmf"
    (is (approx= 0.3 (pmf (->Bernoulli 0.3) 1) 1e-6))
    (is (approx= 0.7 (pmf (->Bernoulli 0.3) 0) 1e-6)))
  (testing "mean"
    (is (approx= 0.3 (mean (->Bernoulli 0.3)))))
  (testing "variance"
    (is (approx= (* 0.3 0.7) (variance (->Bernoulli 0.3)) 1e-6)))
  (testing "cdf"
    (is (approx= 0.0 (cdf (->Bernoulli 0.3) -0.5)))
    (is (approx= 0.7 (cdf (->Bernoulli 0.3) 0.5)))
    (is (approx= 1.0 (cdf (->Bernoulli 0.3) 1.5)))))

;; ================================================================
;; Binomial distribution
;; ================================================================

(deftest binomial-test
  (testing "pmf"
    ;; P(X=5; n=10, p=0.5) = C(10,5) * 0.5^10 = 252/1024 ≈ 0.24609
    (is (approx= 0.24609375 (pmf (->Binomial 10 0.5) 5) 1e-4)))
  (testing "mean"
    (is (approx= 5.0 (mean (->Binomial 10 0.5)))))
  (testing "variance"
    (is (approx= 2.5 (variance (->Binomial 10 0.5)))))
  (testing "cdf bounds"
    (is (approx= 0.0 (cdf (->Binomial 10 0.5) -1.0)))
    (is (approx= 1.0 (cdf (->Binomial 10 0.5) 10.0)))))

;; ================================================================
;; Geometric distribution
;; ================================================================

(deftest geometric-test
  (testing "pmf at 0"
    ;; P(X=0; p=0.5) = 0.5
    (is (approx= 0.5 (pmf (->Geometric 0.5) 0) 1e-6)))
  (testing "pmf at k"
    ;; P(X=3; p=0.5) = 0.5 * 0.5^3 = 0.0625
    (is (approx= 0.0625 (pmf (->Geometric 0.5) 3) 1e-6)))
  (testing "mean"
    ;; E[X] = (1-p)/p = 1 for p=0.5
    (is (approx= 1.0 (mean (->Geometric 0.5)))))
  (testing "variance"
    ;; Var = (1-p)/p^2 = 2 for p=0.5
    (is (approx= 2.0 (variance (->Geometric 0.5))))))

;; ================================================================
;; New distribution sampling smoke tests
;; ================================================================

(deftest new-distributions-sampling-test
  (testing "beta samples in (0,1)"
    (let [arr (sample-n (->Beta 2.0 5.0) 100)]
      (is (every? #(and (> % 0.0) (< % 1.0)) arr))))
  (testing "cauchy samples finite"
    (let [arr (sample-n (->Cauchy 0.0 1.0) 100)]
      (is (every? #(Double/isFinite %) arr))))
  (testing "student-t samples finite"
    (let [arr (sample-n (->StudentT 3.0) 100)]
      (is (every? #(Double/isFinite %) arr))))
  (testing "binomial samples in [0,n]"
    (let [arr (sample-n (->Binomial 20 0.3) 100)]
      (is (every? #(and (>= % 0.0) (<= % 20.0)) arr))))
  (testing "geometric samples non-negative"
    (let [arr (sample-n (->Geometric 0.5) 100)]
      (is (every? #(>= % 0.0) arr)))))
