(ns raster.sci.special-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.special :refer [lgamma gamma digamma trigamma
                                        gammainc gammaincc
                                        lbeta beta betainc betaincinv
                                        erf erfc erfinv erfcinv
                                        besselj0 besselj1 bessely0 bessely1
                                        besseli0 besseli1 besselk0 besselk1
                                        sinc logit expit log1pexp logaddexp zeta]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Gamma family
;; ================================================================

(deftest lgamma-test
  (testing "lgamma(5) = ln(4!) = ln(24)"
    (is (approx= 3.178053830347946 (lgamma 5.0)))))

(deftest gamma-test
  (testing "gamma(5) = 4! = 24"
    (is (approx= 24.0 (gamma 5.0)))))

(deftest digamma-test
  (testing "digamma(1) = -Euler-Mascheroni constant"
    (is (approx= -0.5772156649015329 (digamma 1.0)))))

(deftest trigamma-test
  (testing "trigamma(1) = pi^2/6"
    (is (approx= 1.6449340668482264 (trigamma 1.0)))))

(deftest gammainc-test
  (testing "gammainc(1, 1) = 1 - 1/e"
    (is (approx= 0.6321205588285577 (gammainc 1.0 1.0) 1e-4))))

(deftest gammaincc-test
  (testing "gammaincc(1, 1) = 1/e"
    (is (approx= (/ 1.0 Math/E) (gammaincc 1.0 1.0) 1e-4))))

;; ================================================================
;; Beta family
;; ================================================================

(deftest lbeta-test
  (testing "lbeta(2, 3) = ln(B(2,3)) = ln(1! * 2! / 4!) = ln(1/12)"
    (is (approx= (Math/log (/ 1.0 12.0)) (lbeta 2.0 3.0) 1e-4))))

(deftest beta-test
  (testing "beta(2, 3) = 1/12"
    (is (approx= (/ 1.0 12.0) (beta 2.0 3.0) 1e-4))))

(deftest betainc-test
  (testing "betainc(1, 1, 0.5) = 0.5 (uniform distribution CDF)"
    (is (approx= 0.5 (betainc 1.0 1.0 0.5) 1e-4))))

(deftest betaincinv-test
  (testing "betaincinv(1, 1, 0.5) = 0.5 (inverse of uniform)"
    (is (approx= 0.5 (betaincinv 1.0 1.0 0.5) 1e-4))))

;; ================================================================
;; Error functions
;; ================================================================

(deftest erf-zero-test
  (testing "erf(0) = 0"
    (is (approx= 0.0 (erf 0.0)))))

(deftest erf-one-test
  (testing "erf(1) ≈ 0.8427008"
    (is (approx= 0.8427007929497149 (erf 1.0)))))

(deftest erfc-test
  (testing "erfc(0) = 1"
    (is (approx= 1.0 (erfc 0.0))))
  (testing "erfc(1) = 1 - erf(1)"
    (is (approx= (- 1.0 0.8427007929497149) (erfc 1.0)))))

(deftest erfinv-test
  (testing "erfinv(0) = 0"
    (is (approx= 0.0 (erfinv 0.0))))
  (testing "erfinv(erf(1)) ≈ 1.0"
    (is (approx= 1.0 (erfinv 0.8427007929497149) 1e-4))))

(deftest erfcinv-test
  (testing "erfcinv(1) = 0"
    (is (approx= 0.0 (erfcinv 1.0) 1e-4))))

;; ================================================================
;; Bessel functions
;; ================================================================

(deftest besselj0-test
  (testing "besselj0(0) = 1"
    (is (approx= 1.0 (besselj0 0.0))))
  (testing "besselj0 near first zero (2.4048)"
    (is (approx= 0.0 (besselj0 2.4048255577) 1e-4))))

(deftest besselj1-test
  (testing "besselj1(0) = 0"
    (is (approx= 0.0 (besselj1 0.0))))
  (testing "besselj1(1) ≈ 0.44005"
    (is (approx= 0.44005058574493355 (besselj1 1.0)))))

(deftest bessely0-test
  (testing "bessely0(1) is finite and close to reference"
    ;; Y0(1) ≈ 0.08825696
    (is (approx= 0.08825696421567696 (bessely0 1.0) 1e-4))))

(deftest bessely1-test
  (testing "bessely1(1) ≈ -0.78121"
    (is (approx= -0.7812128213002887 (bessely1 1.0) 1e-4))))

(deftest besseli0-test
  (testing "besseli0(0) = 1"
    (is (approx= 1.0 (besseli0 0.0)))))

(deftest besseli1-test
  (testing "besseli1(0) = 0"
    (is (approx= 0.0 (besseli1 0.0) 1e-4))))

(deftest besselk0-test
  (testing "besselk0(1) ≈ 0.42102"
    (is (approx= 0.42102443824070834 (besselk0 1.0) 1e-4))))

(deftest besselk1-test
  (testing "besselk1(1) ≈ 0.60191"
    (is (approx= 0.6019072301972346 (besselk1 1.0) 1e-4))))

;; ================================================================
;; Utility special functions
;; ================================================================

(deftest sinc-test
  (testing "sinc(0) = 1"
    (is (approx= 1.0 (sinc 0.0)))))

(deftest logit-test
  (testing "logit(0.5) = 0"
    (is (approx= 0.0 (logit 0.5)))))

(deftest expit-test
  (testing "expit(0) = 0.5"
    (is (approx= 0.5 (expit 0.0)))))

(deftest log1pexp-test
  (testing "log1pexp(0) = ln(2)"
    (is (approx= (Math/log 2.0) (log1pexp 0.0)))))

(deftest logaddexp-test
  (testing "logaddexp(0, 0) = ln(2)"
    (is (approx= (Math/log 2.0) (logaddexp 0.0 0.0)))))

(deftest zeta-test
  (testing "zeta(2) = pi^2/6"
    (is (approx= 1.6449340668482264 (zeta 2.0) 1e-3))))
