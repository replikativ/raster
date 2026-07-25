(ns raster.linalg.contract
  "Differentiable matmul-shaped tensor contraction, with its reverse-mode rule.

   The A2 milestone of the SOAC-einsum arc: a contraction is differentiable, and — the key
   insight from the Dex/Futhark study — the BACKWARD of a contraction is TWO contractions:

       C[i,j] = Σ_l A[i,l]·B[l,j]           (forward)
       Ā[i,l] = Σ_j C̄[i,j]·B[l,j]  (= C̄·Bᵀ)  (grad wrt A)
       B̄[l,j] = Σ_i A[i,l]·C̄[i,j]  (= Aᵀ·C̄)  (grad wrt B)

   Both gradients are themselves `par/contract` forms, so they lower + ROUTE through the same
   contraction machinery (and, at 2-free/1-contract, the DPAS/dp4a peak leaves) as the forward
   — 'split for AD, re-fuse after'. Because the accumulate is `+`, the reduce adjoint is the
   trivial broadcast and only the elementwise product needs the product rule; no fused hand-VJP
   is required (matching what Dex and Futhark do). Finite-difference-validated to ~1e-9."
  (:require [raster.core :refer [deftm]]
            [raster.par]
            [raster.arrays :as ra]
            [raster.ad.templates :as tmpl]))

;; ── forward + the two backward contractions (all expressed as par/contract) ──────────
(deftm contract-mm
  [A :- (Array double), B :- (Array double), m :- Long, k :- Long, n :- Long] :- (Array double)
  (let [C (ra/alloc-like A (* m n))]
    (raster.par/contract C [[i m] [j n]] [[l k]]
      (* (ra/aget A (+ (* i k) l)) (ra/aget B (+ (* l n) j))))
    C))

(deftm contract-mm-dA
  "Ā[i,l] = Σ_j C̄[i,j]·B[l,j]  (= C̄·Bᵀ)."
  [Cbar :- (Array double), B :- (Array double), m :- Long, k :- Long, n :- Long] :- (Array double)
  (let [dA (ra/alloc-like Cbar (* m k))]
    (raster.par/contract dA [[i m] [l k]] [[j n]]
      (* (ra/aget Cbar (+ (* i n) j)) (ra/aget B (+ (* l n) j))))
    dA))

(deftm contract-mm-dB
  "B̄[l,j] = Σ_i A[i,l]·C̄[i,j]  (= Aᵀ·C̄)."
  [A :- (Array double), Cbar :- (Array double), m :- Long, k :- Long, n :- Long] :- (Array double)
  (let [dB (ra/alloc-like Cbar (* k n))]
    (raster.par/contract dB [[l k] [j n]] [[i m]]
      (* (ra/aget A (+ (* i k) l)) (ra/aget Cbar (+ (* i n) j))))
    dB))

;; ── reverse-mode rule: backward = the two gradient contractions ──────────────────────
(tmpl/merge-into-template! 'raster.linalg.contract/contract-mm
                           {:params '[A B m k n] :result nil :adjoint 'dy
                            :grads-fn (fn [ctx [A B m k n] _result-sym adjoint-sym gensym-fn]
                                        (let [dA (gensym-fn "dA" (tmpl/grad-tag A))
                                              dB (gensym-fn "dB" (tmpl/grad-tag B))]
                                          [(update ctx :bindings into
                                                   [dA (list 'raster.linalg.contract/contract-mm-dA adjoint-sym B m k n)
                                                    dB (list 'raster.linalg.contract/contract-mm-dB A adjoint-sym m k n)])
                                           [dA dB nil nil nil]]))})
