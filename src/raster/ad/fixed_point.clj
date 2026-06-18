(ns raster.ad.fixed-point
  "Fixed-point iteration with implicit differentiation.

   Provides fixed-point-solve for finding z* = g(z*, theta) and an rrule
   that differentiates through the solution using the Implicit Function
   Theorem (IFT), avoiding backpropagation through iterations (O(1) memory).

   Scalar IFT:
     dz*/dtheta = (dg/dtheta) / (1 - dg/dz)

   Derivatives of g are computed via value+grad — no finite differences.

   The rrule is registered so reverse-mode AD differentiates through
   fixed-point-solve calls automatically."
  (:refer-clojure :exclude [+ - * /])
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Forward: scalar fixed-point iteration
;; ================================================================

(deftm fixed-point-solve
  "Iterate z_{k+1} = g(z_k, theta) until |z_{k+1} - z_k| < tol or maxiter.
   g: (Double, Double) -> Double, should be a contraction mapping.
   Returns converged z*."
  [g :- (Fn [Double Double] Double),
   z0 :- Double, theta :- Double,
   tol :- Double, maxiter :- Long] :- Double
  (loop [z z0, iter (int 0)]
    (if (>= iter maxiter)
      z
      (let [z-new (g z theta)]
        (if (< (Math/abs (- z-new z)) tol)
          z-new
          (recur z-new (unchecked-add-int iter 1)))))))

;; ================================================================
;; Backward: scalar IFT via AD
;; ================================================================

(deftm fixed-point-backward
  "IFT backward for scalar fixed-point.

   At converged z*, computes dz*/dtheta using the IFT:
     dz*/dtheta = (dg/dtheta) / (1 - dg/dz)

   Derivatives of g are computed via value+grad (exact, not finite diff).
   Returns v * dz*/dtheta (cotangent-weighted)."
  [g :- (Fn [Double Double] Double),
   z-star :- Double, theta :- Double, v :- Double] :- Double
  (let [;; Compute dg/dz and dg/dtheta at (z*, theta) via value+grad
        ;; value+grad of g w.r.t. both args: [g(z,theta), dg/dz, dg/dtheta]
        vg-fn (raster.ad.reverse/value+grad g)
        vg-result (vg-fn z-star theta)
        dgdz (nth vg-result 1)
        dgdtheta (nth vg-result 2)
        ;; IFT: dz*/dtheta = dgdtheta / (1 - dgdz)
        denom (- 1.0 dgdz)]
    (if (< (Math/abs denom) 1e-15)
      0.0
      (* v (/ dgdtheta denom)))))

;; ================================================================
;; Rrule: register so AD can differentiate through fixed-point-solve
;; ================================================================

(tmpl/merge-into-template! 'raster.ad.fixed-point/fixed-point-solve
                           {:pullback-factory (fn [z-star g z0 theta tol maxiter]
                       ;; Pullback: given cotangent v (adjoint of z*), compute gradients
                       ;; Only theta gets a gradient via IFT; g, z0, tol, maxiter get nil
                                                (fn [v]
                                                  [nil nil (fixed-point-backward g z-star theta v) nil nil]))})

;; ================================================================
;; Template registration for fixed-point-solve
;; ================================================================

(tmpl/merge-into-template! 'raster.ad.fixed-point/fixed-point-solve
                           {:params '[g z0 theta tol maxiter] :result 'z-star :adjoint 'v
                            :grads-fn (fn [ctx [g z0 theta tol maxiter] result-sym adjoint-sym gensym-fn]
                                        (let [d-theta (gensym-fn "d_theta")]
                                          [(update ctx :bindings into
                                                   [d-theta (list 'raster.ad.fixed-point/fixed-point-backward
                                                                  g result-sym theta adjoint-sym)])
                                           [nil nil d-theta nil nil]]))})
