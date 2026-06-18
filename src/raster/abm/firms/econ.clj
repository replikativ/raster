(ns raster.abm.firms.econ
  "Pure economic functions for the Firms ABM.

   Production: Y = a*E + b*E^β (Cobb-Douglas with linear term)
   Utility:    U = income^θ * (endow - e)^(1-θ) (Cobb-Douglas)
   Effort:     Newton-Raphson FOC solver for equal-shares compensation

   All functions are deftm — zero allocation, primitive fast-path."
  (:refer-clojure :exclude [+ - * /])
  (:require [raster.core :refer [deftm]]
            [raster.numeric :as n :refer [+ - * /]]))

;; ================================================================
;; Production function: Y = a*E + b*E^β
;; ================================================================

(deftm production [a :- Double, b :- Double, beta :- Double,
                   total-effort :- Double] :- Double
  (+ (* a total-effort) (* b (Math/pow total-effort beta))))

;; ================================================================
;; Marginal product: dY/dE = a + b*β*E^(β-1)
;; ================================================================

(deftm marginal-product [a :- Double, b :- Double, beta :- Double,
                         total-effort :- Double] :- Double
  (+ a (* b beta (Math/pow total-effort (- beta 1.0)))))

;; ================================================================
;; Log utility: θ*ln(income) + (1-θ)*ln(endow - effort)
;; ================================================================

(deftm log-utility [theta :- Double, income :- Double,
                    endowment :- Double, effort :- Double] :- Double
  (let [leisure (- endowment effort)]
    (if (> leisure 1e-300)
      (+ (* theta (Math/log income))
         (* (- 1.0 theta) (Math/log leisure)))
      (* theta (Math/log income)))))

;; ================================================================
;; Wage computation — equal shares: Y/n
;; ================================================================

(deftm wage-equal-shares [output :- Double, n-workers :- Long] :- Double
  (/ output (double n-workers)))

;; ================================================================
;; Wage computation — proportional: Y*(e_i/E)
;; ================================================================

(deftm wage-proportional [output :- Double, effort-i :- Double,
                          total-effort :- Double] :- Double
  (if (> total-effort 0.0)
    (* output (/ effort-i total-effort))
    0.0))

;; ================================================================
;; Effort solver (Newton-Raphson on FOC) — equal shares compensation
;;
;; Maximizes V(e) = θ*ln(Y) + (1-θ)*ln(ω-e)   (N drops out of FOC)
;; where Y = a*E + b*E^β, E = e + others-effort
;;
;; FOC:   f(e)  = θ*MP/Y - (1-θ)/(ω-e) = 0
;; Deriv: f'(e) = θ*(dMP*Y - MP²)/Y² - (1-θ)/(ω-e)²
;;
;; with MP = a + b*β*E^(β-1), dMP = b*β*(β-1)*E^(β-2)
;;
;; Works for both singleton (others-effort=0) and team cases.
;; ================================================================

(deftm solve-effort [a :- Double, b :- Double, beta :- Double,
                     others-effort :- Double, theta :- Double,
                     endowment :- Double] :- Double
  (let [e-min 0.001
        e-max (- endowment 0.001)]
    (loop [guess (* theta endowment)  ;; initial guess: θ*ω
           iter  (int 0)]
      (if (>= iter 10)
        ;; max iters — clamp and return (C11 uses 10 iters, tol 0.001)
        (Math/max e-min (Math/min e-max guess))
        (let [e     (Math/max e-min (Math/min e-max guess))
              E     (+ e others-effort)
              ;; Single pow call: E^(β-2), derive E^(β-1) and E^β via multiply
              Ebm2  (Math/pow E (- beta 2.0))
              Ebm1  (* Ebm2 E)          ;; E^(β-1) = E^(β-2) * E
              Eb    (* Ebm1 E)          ;; E^β     = E^(β-1) * E
              Y     (+ (* a E) (* b Eb))
              MP    (+ a (* b beta Ebm1))
              dMP   (* b beta (- beta 1.0) Ebm2)
              leis  (- endowment e)
              ;; f(e) = θ*MP/Y - (1-θ)/leis
              fe    (- (/ (* theta MP) Y)
                       (/ (- 1.0 theta) leis))
              ;; f'(e) = θ*(dMP*Y - MP²)/Y² - (1-θ)/leis²
              fpe   (- (/ (* theta (- (* dMP Y) (* MP MP)))
                          (* Y Y))
                       (/ (- 1.0 theta)
                          (* leis leis)))]
          (if (or (< (Math/abs fe) 1e-6)
                  (== fpe 0.0))
            ;; converged or stuck
            e
            (let [step     (/ fe fpe)
                  ;; damped step to avoid overshooting
                  step     (Math/max (- (/ e-max 2.0)) (Math/min (/ e-max 2.0) step))
                  new-e    (- e step)]
              (if (< (Math/abs step) 0.001)
                (Math/max e-min (Math/min e-max new-e))
                (recur new-e (unchecked-add-int iter 1))))))))))

;; ================================================================
;; Convenience: singleton effort (others-effort = 0)
;; ================================================================

(deftm solve-singleton-effort [a :- Double, b :- Double, beta :- Double,
                               theta :- Double, endowment :- Double] :- Double
  (solve-effort a b beta 0.0 theta endowment))

;; ================================================================
;; Fused solve-effort + utility — avoids recomputing production.
;; Returns effort in result[0], log-utility in result[1].
;; ================================================================

(deftm solve-effort-and-utility!
  [a :- Double, b :- Double, beta :- Double,
   others-effort :- Double, n-workers :- Long,
   theta :- Double, endowment :- Double,
   result :- (Array double)] :- Long
  (let [e (solve-effort a b beta others-effort theta endowment)
        E (+ e others-effort)
        ;; Reuse production from final NR state (one pow call)
        Ebm2 (Math/pow E (- beta 2.0))
        Y    (+ (* a E) (* b (* (* Ebm2 E) E)))
        wage (/ Y (double n-workers))
        leis (- endowment e)
        u    (if (> leis 1e-300)
               (+ (* theta (Math/log wage))
                  (* (- 1.0 theta) (Math/log leis)))
               (* theta (Math/log wage)))]
    (aset result 0 e)
    (aset result 1 u)
    0))

;; ================================================================
;; Prospective log utility — what agent would get at a firm
;; with given effort, under equal shares compensation
;; ================================================================

(deftm prospective-log-utility [a :- Double, b :- Double, beta :- Double,
                                effort :- Double, others-effort :- Double,
                                n-workers :- Long, theta :- Double,
                                endowment :- Double] :- Double
  (let [E      (+ effort others-effort)
        output (production a b beta E)
        wage   (wage-equal-shares output n-workers)]
    (log-utility theta wage endowment effort)))
