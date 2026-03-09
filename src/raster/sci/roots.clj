(ns raster.sci.roots
  "Root finding algorithms following Julia's Roots.jl patterns.

  Provides bracket-based (bisection, Brent) and derivative-based
  (Newton) root finding, plus fixed-point iteration.

  All functions accept typed function arguments via (Fn [Double] Double)
  for primitive-speed evaluation without boxing.

  Usage:
    (require '[raster.sci.roots :refer [bisect newton brent fixed-point]])
    (require '[raster.core :refer [ftm]])
    (bisect (ftm [x :- Double] :- Double (- (* x x) 2.0)) 0.0 2.0)
    (newton (ftm [x :- Double] :- Double (- (* x x) 2.0))
            (ftm [x :- Double] :- Double (* 2.0 x))
            1.5)"
  (:require [raster.core :refer [deftm ftm]]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; Bisection method
;; ================================================================

(deftm bisect (All [T] [f :- (Fn [T] T),
                        a :- T, b :- T,
                        tol :- T, maxiter :- Long]
                   (let [fa (f a)
                         fb (f b)]
                     (when (>= (* fa fb) 0.0)
                       (throw (ex-info "f(a) and f(b) must have opposite signs"
                                       {:a a :b b :fa fa :fb fb})))
                     (loop [a  a
                            b  b
                            fa fa
                            fb fb
                            iter (int 0)]
                       (let [mid (/ (+ a b) 2.0)
                             fm  (f mid)]
                         (if (or (>= iter maxiter)
                                 (< (n/abs (- b a)) tol)
                                 (< (n/abs fm) tol))
                           {:root mid :f-root fm :iterations (long iter)
                            :converged? (or (< (n/abs (- b a)) tol)
                                            (< (n/abs fm) tol))}
                           (if (< (* fa fm) 0.0)
                             (recur a mid fa fm (unchecked-add-int iter 1))
                             (recur mid b fm fb (unchecked-add-int iter 1)))))))))

;; 3-arg convenience: default tol=1e-12, maxiter=100
(deftm bisect [f :- (Fn [Double] Double), a :- Double, b :- Double]
  (bisect f a b 1e-12 100))

;; ================================================================
;; Newton's method for scalar root finding
;; ================================================================

(deftm newton (All [T] [f :- (Fn [T] T),
                        df :- (Fn [T] T),
                        x0 :- T,
                        tol :- T, maxiter :- Long]
                   (loop [x    x0
                          iter (int 0)]
                     (let [fx  (f x)
                           dfx (df x)]
                       (if (or (>= iter maxiter)
                               (< (n/abs fx) tol))
                         {:root x :f-root fx :iterations (long iter)
                          :converged? (< (n/abs fx) tol)}
                         (if (< (n/abs dfx) 1e-15)
                           {:root x :f-root fx :iterations (long iter) :converged? false}
                           (recur (- x (/ fx dfx)) (unchecked-add-int iter 1))))))))

;; 3-arg convenience: default tol=1e-12, maxiter=50
(deftm newton [f :- (Fn [Double] Double),
               df :- (Fn [Double] Double),
               x0 :- Double]
  (newton f df x0 1e-12 50))

;; Convenience: Newton with finite-difference derivative

(deftm newton-fd [f :- (Fn [Double] Double),
                  x0 :- Double,
                  tol :- Double, maxiter :- Long]
  (loop [x    x0
         iter (int 0)]
    (let [fx  (f x)
          eps 1e-8
          dfx (/ (- (f (+ x eps)) (f (- x eps))) (* 2.0 eps))]
      (if (or (>= iter maxiter)
              (< (n/abs fx) tol))
        {:root x :f-root fx :iterations (long iter)
         :converged? (< (n/abs fx) tol)}
        (if (< (n/abs dfx) 1e-15)
          {:root x :f-root fx :iterations (long iter) :converged? false}
          (recur (- x (/ fx dfx)) (unchecked-add-int iter 1)))))))

;; 2-arg convenience: default tol=1e-12, maxiter=50
(deftm newton-fd [f :- (Fn [Double] Double), x0 :- Double]
  (newton-fd f x0 1e-12 50))

;; ================================================================
;; Brent's method — Brent-Dekker algorithm
;; a is the contrapoint, b is the best approximation.
;; ================================================================

(deftm brent (All [T] [f :- (Fn [T] T),
                       a0 :- T, b0 :- T,
                       tol :- T, maxiter :- Long]
                  (let [fa0 (f a0)
                        fb0 (f b0)]
                    (when (>= (* fa0 fb0) 0.0)
                      (throw (ex-info "f(a) and f(b) must have opposite signs"
                                      {:a a0 :b b0 :fa fa0 :fb fb0})))
    ;; Ensure |f(b)| <= |f(a)|
                    (let [[a b fa fb] (if (< (n/abs fa0) (n/abs fb0))
                                        [b0 a0 fb0 fa0]
                                        [a0 b0 fa0 fb0])]
                      (loop [a a, b b, c a, d (- b a)
                             fa fa, fb fb, fc fa
                             mflag true
                             iter (int 0)]
                        (if (or (== fb 0.0) (>= iter maxiter)
                                (< (n/abs (- b a)) tol))
                          {:root b :f-root fb :iterations (long iter)
                           :converged? (or (== fb 0.0) (< (n/abs (- b a)) tol))}
          ;; Compute candidate s
                          (let [s (if (and (not (== fa fc)) (not (== fb fc)))
                    ;; Inverse quadratic interpolation
                                    (+ (/ (* a fb fc) (* (- fa fb) (- fa fc)))
                                       (/ (* b fa fc) (* (- fb fa) (- fb fc)))
                                       (/ (* c fa fb) (* (- fc fa) (- fc fb))))
                    ;; Secant method
                                    (- b (/ (* fb (- b a)) (- fb fa))))
                ;; Bisection conditions (reject s if any true)
                                cond-ab (let [lo (/ (+ (* 3.0 a) b) 4.0)]
                                          (not (if (< lo b)
                                                 (and (<= lo s) (<= s b))
                                                 (and (<= b s) (<= s lo)))))
                                cond-bc (and mflag (>= (n/abs (- s b)) (/ (n/abs (- b c)) 2.0)))
                                cond-cd (and (not mflag) (>= (n/abs (- s b)) (/ (n/abs (- c d)) 2.0)))
                                cond-btol (and mflag (< (n/abs (- b c)) tol))
                                cond-dtol (and (not mflag) (< (n/abs (- c d)) tol))
                                use-bisect (or cond-ab cond-bc cond-cd cond-btol cond-dtol)
                                s (if use-bisect (/ (+ a b) 2.0) s)
                                mflag use-bisect
                                fs (f s)
                ;; d = old c, c = old b
                                d c
                                c b
                                fc fb
                ;; Update bracket: maintain f(a)*f(b) < 0
                                [a b fa fb] (if (< (* fa fs) 0.0)
                                              [a s fa fs]
                                              [s b fs fb])
                ;; Ensure |f(b)| <= |f(a)|
                                [a b fa fb] (if (< (n/abs fa) (n/abs fb))
                                              [b a fb fa]
                                              [a b fa fb])]
                            (recur a b c d fa fb fc mflag (unchecked-add-int iter 1)))))))))

;; 3-arg convenience: default tol=1e-12, maxiter=100
(deftm brent [f :- (Fn [Double] Double), a :- Double, b :- Double]
  (brent f a b 1e-12 100))

;; ================================================================
;; Fixed-point iteration
;; ================================================================

(deftm fixed-point (All [T] [g :- (Fn [T] T),
                             x0 :- T,
                             tol :- T, maxiter :- Long]
                        (loop [x    x0
                               iter (int 0)]
                          (let [x-new (g x)
                                diff  (n/abs (- x-new x))]
                            (if (or (>= iter maxiter) (< diff tol))
                              {:root x-new :iterations (long iter)
                               :converged? (< diff tol)}
                              (recur x-new (unchecked-add-int iter 1)))))))

;; 2-arg convenience: default tol=1e-12, maxiter=100
(deftm fixed-point [g :- (Fn [Double] Double), x0 :- Double]
  (fixed-point g x0 1e-12 100))
