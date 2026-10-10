(ns raster.ad.fixed-point
  "Fixed-point iteration with implicit differentiation.

   Provides fixed-point-solve for finding z* = g(z*, theta) and an rrule
   that differentiates through the solution using the Implicit Function
   Theorem (IFT), avoiding backpropagation through iterations (O(1) memory).

   Scalar IFT:
     dz*/dtheta = (dg/dtheta) / (1 - dg/dz)

   Derivatives of g are computed via value+grad — no finite differences.

   The rrule is registered so reverse-mode AD differentiates through
   fixed-point-solve calls automatically. Budget exhaustion is an error, not
   a differentiable best guess. The derivative is IFT at the accepted numerical
   solution, not differentiation of the finite stopping algorithm."
  (:refer-clojure :exclude [+ - * /])
  (:require [raster.core :refer [deftm]]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Forward: scalar fixed-point iteration
;; ================================================================

(deftm fixed-point-solve
  "Iterate z_{k+1} = g(z_k, theta) until |z_{k+1} - z_k| < tol.
   Returns the accepted numerical solution; throws if maxiter is exhausted.
   Zero budget performs no g evaluation and cannot establish convergence.
   tol must be finite and positive; maxiter must be nonnegative.

   g must be pure/deterministic and define a locally isolated differentiable
   fixed-point branch. The stopping test does not prove contraction, uniqueness
   or mathematical convergence. AD uses IFT at the accepted solution, not the
   derivative of the finite iteration/stopping program."
  [g :- (Fn [Double Double] Double),
   z0 :- Double, theta :- Double,
   tol :- Double, maxiter :- Long] :- Double
  (when (or (not (Double/isFinite tol)) (<= tol 0.0) (< maxiter 0))
    (throw (ex-info "Fixed-point solve requires a positive finite tolerance and nonnegative budget"
                    {:reason :invalid-fixed-point-options :tol tol :maxiter maxiter})))
  (loop [z z0, iter (long 0)]
    (if (>= iter maxiter)
      (throw (ex-info "Fixed-point solve exhausted its iteration budget before convergence"
                      {:reason :fixed-point-not-converged :iterate z
                       :iterations iter :maxiter maxiter :tol tol}))
      (let [z-new (g z theta)]
        (when-not (Double/isFinite z-new)
          (throw (ex-info "Fixed-point update produced a nonfinite iterate"
                          {:reason :nonfinite-fixed-point-iterate
                           :iterate z-new :iterations iter})))
        (if (< (Math/abs (- z-new z)) tol)
          z-new
          (recur z-new (unchecked-inc iter)))))))

;; ================================================================
;; Backward: scalar IFT via AD
;; ================================================================

(deftm fixed-point-backward
  "IFT backward for scalar fixed-point.

   At converged z*, computes dz*/dtheta using the IFT:
     dz*/dtheta = (dg/dtheta) / (1 - dg/dz)

   Derivatives of g are computed via value+grad (not finite differences).
   Caller must supply an accepted solution on a locally isolated differentiable
   branch. Returns v * dz*/dtheta (cotangent-weighted). Nonfinite derivatives
   or results fail; |1-dg/dz| < 1e-15 fails as an ill-conditioning guard,
   not as a proof of exact singularity."
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
    (when (or (not (Double/isFinite (double dgdz)))
              (not (Double/isFinite (double dgdtheta)))
              (not (Double/isFinite (double denom))))
      (throw (ex-info "Fixed-point implicit derivative is nonfinite"
                      {:reason :nonfinite-fixed-point-gradient
                       :dgdz dgdz :dgdtheta dgdtheta :denominator denom})))
    (when (< (Math/abs denom) 1e-15)
      (throw (ex-info "Fixed-point implicit derivative is ill-conditioned"
                      {:reason :ill-conditioned-fixed-point
                       :denominator denom :minimum-magnitude 1e-15})))
    (let [gradient (* v (/ dgdtheta denom))]
      (when-not (Double/isFinite (double gradient))
        (throw (ex-info "Fixed-point cotangent result is nonfinite"
                        {:reason :nonfinite-fixed-point-gradient :gradient gradient})))
      gradient)))

;; ================================================================
;; Template registration for fixed-point-solve
;; ================================================================

(tmpl/merge-into-template! 'raster.ad.fixed-point/fixed-point-solve
                           {:params '[g z0 theta tol maxiter] :result 'z-star :adjoint 'v
                            :grads-fn (fn [ctx [g z0 theta tol maxiter] result-sym adjoint-sym gensym-fn]
                                        (let [d-theta (gensym-fn "d_theta" (tmpl/grad-tag theta))]
                                          [(update ctx :bindings into
                                                   [d-theta (list 'raster.ad.fixed-point/fixed-point-backward
                                                                  g result-sym theta adjoint-sym)])
                                           [nil nil d-theta nil nil]]))})
