(ns raster.abm.firms.differentiable
  "Differentiable Firms ABM — softmax decisions and IFT-based gradients.

   Relaxations for differentiability:
   - Fixed firm set (no birth/death — discrete events)
   - Soft switching via softmax over k alternatives instead of argmax
   - Each agent considers only k firms (current + neighbors), not all
   - Double arrays throughout (AD operates on doubles)

   Key components:
   1. solve-effort rrule — analytical IFT for the Newton-Raphson FOC
   2. evaluate-utilities — utility for each alternative
   3. soft-select — softmax-weighted effort/utility selection
   4. differentiable-period — one simulation step
   5. trajectory-loss — multi-period loss for gradient computation"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.abm.firms.econ :as econ]
            [raster.nn :as nn]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; E.1: solve-effort rrule (analytical IFT)
;; ================================================================
;;
;; At converged e*, FOC holds: F(e*, a, b, β, others_e, θ, ω) = 0
;;   F = θ*MP/Y - (1-θ)/(ω-e)
;; where MP = a + b*β*E^(β-1), Y = a*E + b*E^β, E = e + others_e
;;
;; IFT: de*/dparam_i = -(∂F/∂param_i) / (∂F/∂e)
;; ∂F/∂e = f'(e) is the FOC derivative (already in the NR iteration)
;;
;; We need partials ∂F/∂a, ∂F/∂b, ∂F/∂β, ∂F/∂θ, ∂F/∂ω, ∂F/∂others_e
;;
;; F = θ*MP/Y - (1-θ)/leis
;; MP = a + b*β*E^(β-1)
;; Y  = a*E + b*E^β
;; leis = ω - e

(deftm solve-effort-foc-partials
  "Compute FOC partial derivatives at converged e*.
   Returns array [∂F/∂e, ∂F/∂a, ∂F/∂b, ∂F/∂beta, ∂F/∂others_e, ∂F/∂theta, ∂F/∂omega]."
  [e :- Double, a :- Double, b :- Double, beta :- Double,
   others-effort :- Double, theta :- Double,
   endowment :- Double] :- (Array double)
  (let [result (double-array 7)
        E     (+ e others-effort)
        Ebm2  (Math/pow E (- beta 2.0))
        Ebm1  (* Ebm2 E)
        Eb    (* Ebm1 E)
        Y     (+ (* a E) (* b Eb))
        MP    (+ a (* b beta Ebm1))
        dMP   (* b beta (- beta 1.0) Ebm2)
        leis  (- endowment e)
        Y2    (* Y Y)
        leis2 (* leis leis)
        ;; ∂F/∂e = θ*(dMP*Y - MP²)/Y² - (1-θ)/leis²
        dFde  (- (/ (* theta (- (* dMP Y) (* MP MP))) Y2)
                 (/ (- 1.0 theta) leis2))
        ;; ∂F/∂a: MP has ∂MP/∂a = 1, Y has ∂Y/∂a = E
        ;; ∂F/∂a = θ * (1*Y - MP*E) / Y²
        dFda  (/ (* theta (- Y (* MP E))) Y2)
        ;; ∂F/∂b: ∂MP/∂b = β*E^(β-1), ∂Y/∂b = E^β
        ;; ∂F/∂b = θ * (β*E^(β-1)*Y - MP*E^β) / Y²
        dFdb  (/ (* theta (- (* beta Ebm1 Y) (* MP Eb))) Y2)
        ;; ∂F/∂β: ∂MP/∂β = b*E^(β-1) + b*β*E^(β-1)*ln(E)
        ;;         ∂Y/∂β = b*E^β*ln(E)
        lnE   (Math/log (Math/max E 1e-300))
        dMPdbeta (+ (* b Ebm1) (* b beta Ebm1 lnE))
        dYdbeta  (* b Eb lnE)
        dFdbeta  (/ (* theta (- (* dMPdbeta Y) (* MP dYdbeta))) Y2)
        ;; ∂F/∂others_e: same as ∂F/∂e (E = e + others_e, symmetric)
        ;; Actually E changes by 1 for both, but leis doesn't change for others_e
        ;; ∂F/∂others_e = θ*(dMP*Y - MP²)/Y² (no leisure term)
        dFdoe (/ (* theta (- (* dMP Y) (* MP MP))) Y2)
        ;; ∂F/∂θ: θ appears linearly
        ;; F = θ*MP/Y - (1-θ)/leis = θ*MP/Y - 1/leis + θ/leis
        ;; ∂F/∂θ = MP/Y + 1/leis
        dFdtheta (+ (/ MP Y) (/ 1.0 leis))
        ;; ∂F/∂ω: leis = ω - e, ∂leis/∂ω = 1
        ;; ∂F/∂ω = -(1-θ) * (-1/leis²) = (1-θ)/leis²
        dFdomega (/ (- 1.0 theta) leis2)]
    (aset result 0 dFde)
    (aset result 1 dFda)
    (aset result 2 dFdb)
    (aset result 3 dFdbeta)
    (aset result 4 dFdoe)
    (aset result 5 dFdtheta)
    (aset result 6 dFdomega)
    result))

;; solve-effort's gradient is the implicit-function-theorem pullback (not a
;; static grads-fn — the FOC is solved numerically), so this runtime pullback
;; is its sole gradient representation.
(tmpl/merge-into-template! 'raster.abm.firms.econ/solve-effort
                           {:pullback-factory (fn [e-star a b beta others-effort theta endowment]
                                                (let [partials (solve-effort-foc-partials
                                                                (double e-star) (double a) (double b) (double beta)
                                                                (double others-effort) (double theta) (double endowment))
                                                      dFde     (clojure.core/aget ^doubles partials 0)
                                                      dFda     (clojure.core/aget ^doubles partials 1)
                                                      dFdb     (clojure.core/aget ^doubles partials 2)
                                                      dFdbeta  (clojure.core/aget ^doubles partials 3)
                                                      dFdoe    (clojure.core/aget ^doubles partials 4)
                                                      dFdtheta (clojure.core/aget ^doubles partials 5)
                                                      dFdomega (clojure.core/aget ^doubles partials 6)]
                                                  (fn [v]
                           ;; IFT: de*/dp = -(∂F/∂p) / (∂F/∂e)
                                                    (if (< (Math/abs dFde) 1e-15)
                                                      [0.0 0.0 0.0 0.0 0.0 0.0]
                                                      (let [scale (- (/ (double v) dFde))]
                                                        [(* scale dFda)
                                                         (* scale dFdb)
                                                         (* scale dFdbeta)
                                                         (* scale dFdoe)
                                                         (* scale dFdtheta)
                                                         (* scale dFdomega)])))))})

;; ================================================================
;; E.2: Soft decision selection
;; ================================================================

(deftm evaluate-utilities
  "Compute utility for k alternatives (current firm + k-1 neighbors).
   Returns double-array of utilities.
   alternatives[i] = firm index for i-th option.
   efforts[i] = optimal effort for i-th option."
  [theta :- Double, endowment :- Double,
   efforts :- (Array double),
   incomes :- (Array double),
   k :- Long] :- (Array double)
  (let [utilities (double-array k)]
    (dotimes [i k]
      (let [income (aget incomes i)
            effort (aget efforts i)
            leis   (- endowment effort)]
        (aset utilities i
              (if (> leis 1e-300)
                (+ (* theta (Math/log (Math/max 1e-300 income)))
                   (* (- 1.0 theta) (Math/log leis)))
                (* theta (Math/log (Math/max 1e-300 income)))))))
    utilities))

(deftm soft-select
  "Softmax selection over alternatives with temperature tau.
   Returns weighted-average effort as the 'soft' decision.
   utilities: utility of each alternative
   efforts: effort at each alternative
   tau: temperature (lower = closer to argmax)"
  [utilities :- (Array double), efforts :- (Array double),
   tau :- Double] :- Double
  (let [k (alength utilities)
        scaled (double-array k)]
    ;; Scale utilities by 1/tau
    (dotimes [i k]
      (aset scaled i (/ (aget utilities i) tau)))
    ;; Softmax
    (let [probs (nn/softmax scaled)]
      ;; Weighted sum of efforts
      (loop [i (int 0) acc 0.0]
        (if (>= i k)
          acc
          (recur (unchecked-add-int i 1)
                 (+ acc (* (aget probs i) (aget efforts i)))))))))

(deftm soft-select-index
  "Softmax selection returning index probabilities.
   Returns the probability-weighted index (soft argmax).
   Useful for soft firm selection."
  [utilities :- (Array double), tau :- Double] :- (Array double)
  (let [k (alength utilities)
        scaled (double-array k)]
    (dotimes [i k]
      (aset scaled i (/ (aget utilities i) tau)))
    (nn/softmax scaled)))

;; ================================================================
;; F.1: Single differentiable period
;; ================================================================

(deftm compute-firm-total-effort
  "Sum efforts assigned to each firm. Flat assignment: agent i -> firm assignment[i]."
  [efforts :- (Array double), assignment :- (Array int),
   n-agents :- Long, n-firms :- Long] :- (Array double)
  (let [total (double-array n-firms)]
    (dotimes [i n-agents]
      (let [fi (aget assignment i)
            e  (aget efforts i)]
        (when (>= fi 0)
          (aset total fi (+ (aget total fi) e)))))
    total))

(deftm compute-firm-sizes
  "Count agents per firm."
  [assignment :- (Array int), n-agents :- Long,
   n-firms :- Long] :- (Array int)
  (let [sizes (int-array n-firms)]
    (dotimes [i n-agents]
      (let [fi (aget assignment i)]
        (when (>= fi 0)
          (aset sizes fi (int (inc (aget sizes fi)))))))
    sizes))

(deftm agent-utility-at-firm
  "Compute utility agent i would get at firm fi.
   Uses equal-shares wage."
  [effort-i :- Double, others-effort :- Double,
   n-workers :- Long,
   a :- Double, b :- Double, beta :- Double,
   theta :- Double, endowment :- Double] :- Double
  (let [E    (+ effort-i others-effort)
        Y    (econ/production a b beta E)
        wage (/ Y (double n-workers))
        leis (- endowment effort-i)]
    (if (> leis 1e-300)
      (+ (* theta (Math/log (Math/max 1e-300 wage)))
         (* (- 1.0 theta) (Math/log leis)))
      (* theta (Math/log (Math/max 1e-300 wage))))))

(deftm differentiable-period
  "One differentiable simulation period.
   Each agent considers k alternatives (current firm + k-1 neighbors).
   Decisions are soft (softmax) rather than hard (argmax).
   Returns mean utility (scalar loss).

   Parameters:
   - theta: agent consumption preferences [n-agents]
   - endowment: agent endowments [n-agents]
   - param-a, param-b, param-beta: firm production params [n-firms]
   - efforts: current efforts [n-agents]
   - assignment: current firm assignment [n-agents]
   - neighbors: flat [n-agents * (k-1)] neighbor agent indices
   - n-agents, n-firms, k: dimensions
   - tau: softmax temperature"
  [theta :- (Array double), endowment :- (Array double),
   param-a :- (Array double), param-b :- (Array double),
   param-beta :- (Array double),
   efforts :- (Array double),
   assignment :- (Array int),
   neighbors :- (Array int),
   n-agents :- Long, n-firms :- Long, k :- Long,
   tau :- Double] :- Double
  (let [;; 1. Compute total effort per firm
        total-effort (compute-firm-total-effort efforts assignment n-agents n-firms)
        firm-sizes   (compute-firm-sizes assignment n-agents n-firms)
        ;; 2. Per agent: evaluate alternatives + soft-select new effort
        new-efforts  (double-array n-agents)]
    (dotimes [agent-i n-agents]
      (let [th    (aget theta agent-i)
            end   (aget endowment agent-i)
            cur-fi (aget assignment agent-i)
            old-e  (aget efforts agent-i)
            ;; Build k alternatives: [current-firm, neighbor-firms...]
            alt-efforts  (double-array k)
            alt-incomes  (double-array k)
            ;; Current firm
            cur-a    (aget param-a cur-fi)
            cur-b    (aget param-b cur-fi)
            cur-beta (aget param-beta cur-fi)
            cur-size (aget firm-sizes cur-fi)
            cur-te   (aget total-effort cur-fi)
            cur-oe   (Math/max (- cur-te old-e) 0.0)
            cur-opt-e (econ/solve-effort cur-a cur-b cur-beta cur-oe th end)
            cur-E    (+ cur-opt-e cur-oe)
            cur-Y    (econ/production cur-a cur-b cur-beta cur-E)
            cur-wage (/ cur-Y (Math/max 1.0 (double cur-size)))]
        (aset alt-efforts 0 cur-opt-e)
        (aset alt-incomes 0 cur-wage)
        ;; Neighbor firms
        (let [nb-base (* agent-i (- k 1))]
          (dotimes [j (- k 1)]
            (let [nb-agent (aget neighbors (+ nb-base j))
                  nb-fi    (aget assignment nb-agent)
                  nb-a     (aget param-a nb-fi)
                  nb-b     (aget param-b nb-fi)
                  nb-beta  (aget param-beta nb-fi)
                  nb-size  (aget firm-sizes nb-fi)
                  nb-te    (aget total-effort nb-fi)
                  ;; If I join, estimate my effort using average
                  avg-e    (if (> nb-size 0)
                             (/ nb-te (double nb-size))
                             0.5)
                  est-e    (Math/max 0.001 (Math/min (- end 0.001) avg-e))
                  ;; Wage if I join: Y / (N+1)
                  est-E    (+ nb-te est-e)
                  est-Y    (econ/production nb-a nb-b nb-beta est-E)
                  est-wage (/ est-Y (double (inc nb-size)))]
              (aset alt-efforts (inc j) est-e)
              (aset alt-incomes (inc j) est-wage))))
        ;; Evaluate utilities and soft-select
        (let [utils (evaluate-utilities th end alt-efforts alt-incomes k)
              soft-e (soft-select utils alt-efforts tau)]
          (aset new-efforts agent-i soft-e))))
    ;; Return mean utility with new efforts
    (let [new-total (compute-firm-total-effort new-efforts assignment n-agents n-firms)]
      (loop [i (int 0) sum-u 0.0]
        (if (>= i n-agents)
          (/ sum-u (double n-agents))
          (let [fi    (aget assignment i)
                e     (aget new-efforts i)
                te    (aget new-total fi)
                a     (aget param-a fi)
                b-    (aget param-b fi)
                beta  (aget param-beta fi)
                sz    (aget firm-sizes fi)
                Y     (econ/production a b- beta te)
                wage  (/ Y (Math/max 1.0 (double sz)))
                th    (aget theta i)
                end   (aget endowment i)
                leis  (- end e)
                u     (if (> leis 1e-300)
                        (+ (* th (Math/log (Math/max 1e-300 wage)))
                           (* (- 1.0 th) (Math/log leis)))
                        (* th (Math/log (Math/max 1e-300 wage))))]
            (recur (unchecked-add-int i 1) (+ sum-u u))))))))

;; ================================================================
;; F.2: Multi-period trajectory loss
;; ================================================================

(deftm trajectory-loss
  "Run n-periods of differentiable simulation, return cumulative
   mean utility as scalar loss (to maximize).

   The loop/recur AD from rad.clj handles backward through time."
  [theta :- (Array double), endowment :- (Array double),
   param-a :- (Array double), param-b :- (Array double),
   param-beta :- (Array double),
   init-efforts :- (Array double),
   assignment :- (Array int),
   neighbors :- (Array int),
   n-agents :- Long, n-firms :- Long, k :- Long,
   tau :- Double, n-periods :- Long] :- Double
  (let [efforts (double-array n-agents)]
    ;; Copy initial efforts
    (dotimes [i n-agents]
      (aset efforts i (aget init-efforts i)))
    ;; Run periods
    (loop [t (int 0) total-loss 0.0]
      (if (>= t n-periods)
        (/ total-loss (double n-periods))
        (let [period-u (differentiable-period
                        theta endowment param-a param-b param-beta
                        efforts assignment neighbors
                        n-agents n-firms k tau)]
          (recur (unchecked-add-int t 1) (+ total-loss period-u)))))))
