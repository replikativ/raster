(ns raster.ad
  "Automatic differentiation — public entry point.

  Provides forward-mode (dual numbers), reverse-mode (VJP/closures-as-tape),
  and higher-order derivatives (jets).

  Usage:
    (require '[raster.ad :as ad])
    (ad/derivative (ftm [x :- Double] :- Double (* x x x)) 2.0)
    ;; => 12.0

  Sub-namespaces for advanced use:
    raster.ad.forward   — dual number forward-mode AD
    raster.ad.reverse   — reverse-mode AD (VJP, gradient, Hessian)
        raster.ad.jet       — truncated Taylor series (higher-order)
        raster.ad.pullbacks — runtime pullback factories
    raster.ad.activity  — activity analysis
    raster.ad.purity    — purity analysis for AD safety"
  (:require [raster.ad.forward]
            [raster.ad.reverse]
            [raster.ad.jet]
            [raster.ad.templates]
            [raster.ad.pullbacks]
            [raster.support :refer [import-vars]]))

;; ================================================================
;; Forward-mode AD
;; ================================================================

(import-vars raster.ad.forward
             dual derivative gradient jacobian
             stop-gradient straight-through)

;; ================================================================
;; Reverse-mode AD
;; ================================================================

(import-vars raster.ad.reverse
             vjp
             compile-hvp-fn compile-hessian-fn
             numerical-gradient
             value+grad grad)

;; ================================================================
;; Jets (higher-order derivatives)
;; ================================================================

(import-vars raster.ad.jet
             jet jet-const jet-from-coeffs
             taylor-coefficient taylor-derivative
             primal higher-derivatives)

;; ================================================================
;; AD templates (gradient rules)
;; ================================================================

(import-vars raster.ad.templates
             register-template! get-pullback-factory has-ad-rule?)

