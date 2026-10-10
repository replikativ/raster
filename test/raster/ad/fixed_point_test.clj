(ns raster.ad.fixed-point-test
  "Tests for fixed-point solver with IFT-based implicit differentiation."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.ad.fixed-point :as fp]))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

;; ================================================================
;; g functions for testing
;; ================================================================

(def sqrt-g
  "Newton iteration for sqrt: g(z,theta) = (z + theta/z) / 2"
  (ftm [z :- Double, theta :- Double] :- Double
       (/ (+ z (/ theta z)) 2.0)))

;; Contraction using only AD-differentiable ops: g(z,theta) = theta * z / (1 + z^2)
;; This is a contraction for |theta| < 1
(def contraction-g
  (ftm [z :- Double, theta :- Double] :- Double
       (n/* theta (n// z (n/+ 1.0 (n/* z z))))))

(def affine-g (ftm [z :- Double theta :- Double] :- Double
                  (n/+ (n/* 0.5 z) theta)))
(def singular-g (ftm [z :- Double theta :- Double] :- Double (n/+ z theta)))
(def near-singular-g
  (ftm [z :- Double theta :- Double] :- Double
       (n/+ (n/* 0.9999999999999995 z) theta)))
(def nonfinite-derivative-g
  (ftm [z :- Double theta :- Double] :- Double (n/+ (Math/sqrt z) theta)))

(defn- error-of [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error error)))

(deftm budgeted-affine [z0 :- Double theta :- Double budget :- Long] :- Double
  (fp/fixed-point-solve affine-g z0 theta 1e-12 budget))

(deftm singular-solution [theta :- Double] :- Double
  (fp/fixed-point-solve singular-g 0.0 theta 1e-12 (long 1)))

;; ================================================================
;; Forward solve tests
;; ================================================================

(deftest fixed-point-sqrt-test
  (testing "fixed-point-solve finds sqrt(theta)"
    (doseq [theta [1.0 2.0 4.0 9.0 16.0 100.0]]
      (let [z* (fp/fixed-point-solve sqrt-g 1.0 theta 1e-12 100)]
        (is (approx= z* (Math/sqrt theta))
            (str "sqrt(" theta ") = " z*))))))

(deftest fixed-point-contraction-test
  (testing "contraction mapping converges"
    (let [theta 0.5
          z* (fp/fixed-point-solve contraction-g 0.3 theta 1e-12 200)]
      ;; z* should satisfy z* = theta * z* / (1 + z*^2)
      (is (approx= z* (n/* theta (n// z* (n/+ 1.0 (n/* z* z*)))) 1e-6)
          "Fixed point residual should be near zero"))))

(deftest fixed-point-budget-is-not-a-convergence-witness-test
  (doseq [budget [0 1 3]]
    (let [updates (atom 0)
          ;; Effectful diagnostic fixture counts evaluations only; it is not
          ;; submitted to AD and is outside the pure mathematical-g premise.
          g (ftm [z :- Double theta :- Double] :- Double
                 (do (swap! updates inc) (n/+ (n/* 0.5 z) theta)))
          error (error-of #(fp/fixed-point-solve g 3.0 1.0 1e-20 budget))]
      (is (= :fixed-point-not-converged (:reason (ex-data error))))
      (is (= budget (:iterations (ex-data error))))
      (is (= budget @updates) "Exhaustion does not evaluate g an extra time")))
  (is (= :fixed-point-not-converged
         (:reason (ex-data (error-of #(fp/fixed-point-solve affine-g 2.0 1.0 1e-12 0)))))
      "An initially fixed point still needs an update to establish the stopping test")
  (is (= 2.0 (fp/fixed-point-solve affine-g 2.0 1.0 1e-12 1))
      "Convergence on the final permitted update succeeds")
  (is (= 2.5 (fp/fixed-point-solve affine-g 3.0 1.0 0.6 1))
      "A nonzero accepted update on the final iteration also succeeds"))

(deftest fixed-point-options-and-nonfinite-iterates-test
  (doseq [[tol budget] [[0.0 1] [-1.0 1] [Double/NaN 1]
                        [Double/POSITIVE_INFINITY 1] [1e-12 -1]]]
    (is (= :invalid-fixed-point-options
           (:reason (ex-data (error-of #(fp/fixed-point-solve affine-g 3.0 1.0 tol budget)))))))
  (is (= :nonfinite-fixed-point-iterate
         (:reason (ex-data (error-of #(fp/fixed-point-solve
                                      (ftm [z :- Double theta :- Double] :- Double Double/NaN)
                                      0.0 0.0 1e-12 1)))))))

;; ================================================================
;; IFT backward tests (direct, no rrule)
;; ================================================================

(deftest ift-sqrt-gradient-test
  (testing "IFT gradient matches analytical 1/(2*sqrt(theta))"
    (doseq [theta [1.0 2.0 4.0 9.0 16.0]]
      (let [z* (fp/fixed-point-solve sqrt-g 1.0 theta 1e-12 100)
            grad (fp/fixed-point-backward sqrt-g z* theta 1.0)
            expected (/ 1.0 (* 2.0 (Math/sqrt theta)))]
        (is (approx= grad expected 1e-6)
            (str "d(sqrt(" theta "))/dtheta = " grad))))))

(deftest ift-contraction-gradient-vs-fd-test
  (testing "IFT gradient matches finite difference for contraction"
    (let [theta 0.5
          eps 1e-5
          z*  (fp/fixed-point-solve contraction-g 0.3 theta 1e-12 200)
          z*p (fp/fixed-point-solve contraction-g 0.3 (+ theta eps) 1e-12 200)
          z*m (fp/fixed-point-solve contraction-g 0.3 (- theta eps) 1e-12 200)
          fd-grad (/ (- z*p z*m) (* 2.0 eps))
          ift-grad (fp/fixed-point-backward contraction-g z* theta 1.0)]
      (is (approx= ift-grad fd-grad 1e-3)
          (str "IFT=" ift-grad " FD=" fd-grad)))))

;; ================================================================
;; Rrule: AD differentiates through fixed-point-solve automatically
;; ================================================================

(deftm sqrt-via-fp [theta :- Double] :- Double
  (fp/fixed-point-solve sqrt-g 1.0 theta 1e-12 (long 100)))

(deftm square-of-sqrt-via-fp [theta :- Double] :- Double
  (let [z (fp/fixed-point-solve sqrt-g 1.0 theta 1e-12 (long 100))]
    (n/* z z)))

(deftm sqrt-derivative-via-fp [theta :- Double] :- Double
  (nth ((rev/value+grad #'sqrt-via-fp :mode :reverse) theta) 1))

(deftest registered-sqrt-gradient-test
  (testing "The registered reverse rule gives the accepted solution's IFT gradient"
    (doseq [theta [1.0 4.0 9.0 16.0]]
      (let [[z* grad] ((rev/value+grad #'sqrt-via-fp :mode :reverse) theta)
            expected-grad (/ 1.0 (* 2.0 (Math/sqrt theta)))]
        (is (approx= (Math/sqrt theta) z*)
            (str "sqrt(" theta ") value"))
        (is (approx= grad expected-grad 1e-6)
            (str "d(sqrt(" theta "))/dtheta"))))))

(deftest registered-implicit-rule-composes-test
  (let [gradient (rev/value+grad #'square-of-sqrt-via-fp :mode :reverse)]
    (doseq [theta [2.0 4.0 9.0]]
      (let [[value dtheta] (gradient theta)]
        (is (approx= value theta 1e-10))
        (is (approx= dtheta 1.0 1e-10))))))

(deftest registered-rule-rejects-budgeted-iterates-test
  (let [gradient (rev/value+grad #'budgeted-affine :mode :reverse :wrt [0 1])]
    (doseq [budget [0 1]]
      (is (= :fixed-point-not-converged
             (:reason (ex-data (error-of #(gradient 3.0 1.0 budget)))))))
    (let [[value dz0 dtheta dbudget] (gradient 3.0 1.0 100)]
      (is (approx= value 2.0 1e-11))
      (is (= 0.0 dz0) "IFT does not differentiate initial-state iteration history")
      (is (= 2.0 dtheta))
      (is (nil? dbudget)))))

(deftest implicit-derivative-conditioning-is-explicit-test
  (doseq [g [singular-g near-singular-g]]
    (is (= :ill-conditioned-fixed-point
           (:reason (ex-data (error-of #(fp/fixed-point-backward g 0.0 0.0 1.0)))))))
  (is (= :ill-conditioned-fixed-point
         (:reason (ex-data (error-of #((rev/value+grad #'singular-solution :mode :reverse) 0.0)))))
      "The registered rule must also reject an accepted but non-isolated solution")
  (is (= :nonfinite-fixed-point-gradient
         (:reason (ex-data (error-of #(fp/fixed-point-backward nonfinite-derivative-g 0.0 0.0 1.0))))))
  (is (= :nonfinite-fixed-point-gradient
         (:reason (ex-data (error-of #(fp/fixed-point-backward affine-g 2.0 1.0 Double/MAX_VALUE))))))
  (is (= :nonfinite-fixed-point-gradient
         (:reason (ex-data (error-of #(fp/fixed-point-backward affine-g 2.0 1.0 Double/NaN))))))
  (is (= 6.0 (fp/fixed-point-backward affine-g 2.0 1.0 3.0))))

(deftest rrule-gradient-vs-fd-test
  (testing "IFT gradient matches finite difference of forward solve"
    (let [h 1e-5]
      (doseq [theta [2.0 4.0 9.0]]
        (let [z* (fp/fixed-point-solve sqrt-g 1.0 theta 1e-12 100)
              grad (second ((rev/value+grad #'sqrt-via-fp :mode :reverse) theta))
              fd (/ (- (fp/fixed-point-solve sqrt-g 1.0 (+ theta h) 1e-12 100)
                       (fp/fixed-point-solve sqrt-g 1.0 (- theta h) 1e-12 100))
                    (* 2.0 h))]
          (is (approx= grad fd 1e-4)
              (str "IFT grad matches FD at theta=" theta)))))))
