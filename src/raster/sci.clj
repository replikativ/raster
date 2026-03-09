(ns raster.sci
  "Scientific computing — public entry point.

  Provides special functions, probability distributions, statistics,
  optimization, root finding, numerical integration, interpolation,
  signal processing, and FFT.

  Usage:
    (require '[raster.sci :as sci])
    (sci/erf 1.0)           ;; => 0.8427...
    (sci/bisect f 0.0 2.0)  ;; root finding
    (sci/quadgk f 0.0 1.0)  ;; numerical integration

  Sub-namespaces for advanced use:
    raster.sci.special        — gamma, beta, erf, Bessel functions
    raster.sci.distributions  — Normal, Gamma, Poisson, etc.
    raster.sci.stats          — t-test, chi², KS test, correlation
    raster.sci.optim          — L-BFGS, Nelder-Mead, Newton, Adam, LM
    raster.sci.roots          — bisection, Brent, Newton root finding
    raster.sci.quadrature     — Gauss-Kronrod, Simpson, trapezoidal
    raster.sci.interpolation  — linear, cubic spline, Akima, PCHIP
    raster.sci.signal         — windows, Welch, convolution, FIR filters
    raster.sci.fft            — FFT, IFFT, DCT, DST"
  (:require [raster.sci.special]
            [raster.sci.distributions]
            [raster.sci.stats]
            [raster.sci.optim]
            [raster.sci.roots]
            [raster.sci.quadrature]
            [raster.sci.interpolation]
            [raster.sci.signal]
            [raster.sci.fft]
            [raster.support :refer [import-vars]]))

;; ================================================================
;; Special functions
;; ================================================================

(import-vars raster.sci.special
             gamma lgamma digamma trigamma
             gammainc gammaincc
             beta lbeta betainc betaincinv
             erf erfc erfinv erfcinv
             besselj0 besselj1 bessely0 bessely1
             besseli0 besseli1 besselk0 besselk1
             sinc logit expit)

;; ================================================================
;; Distributions
;; ================================================================

(import-vars raster.sci.distributions
             logpdf pdf logpmf pmf cdf quantile
             mean variance std)

;; ================================================================
;; Statistics
;; ================================================================

(import-vars raster.sci.stats
             array-mean array-var
             t-test-1sample t-test-2sample
             chi2-test f-test ks-test
             pearson spearman covariance
             percentile describe)

;; ================================================================
;; Optimization
;; ================================================================

(import-vars raster.sci.optim
             optimize nelder-mead gradient-descent lbfgs
             newton-optimize levenberg-marquardt gauss-newton
             curve-fit differential-evolution projected-gradient)

;; ================================================================
;; Root finding
;; ================================================================

(import-vars raster.sci.roots
             bisect newton newton-fd brent fixed-point)

;; ================================================================
;; Numerical integration
;; ================================================================

(import-vars raster.sci.quadrature
             quadgk adaptive-quadgk trapz simps)

;; ================================================================
;; Interpolation
;; ================================================================

(import-vars raster.sci.interpolation
             linear-interp cubic-spline akima-spline pchip
             interp2d-linear interp2d-cubic)

;; ================================================================
;; Signal processing
;; ================================================================

(import-vars raster.sci.signal
             hann hamming blackman kaiser
             welch correlate convolve lfilter firwin sosfilt)

;; ================================================================
;; FFT
;; ================================================================

(import-vars raster.sci.fft
             fft ifft rfft fft-complex
             fftfreq magnitude phase
             dct idct dst idst)
