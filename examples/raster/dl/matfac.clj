(ns raster.dl.matfac
  "Matrix factorization training example for Raster pipeline validation.

  Decomposes X ≈ A @ B^T where A:[M,r], B:[N,r].
  Uses compile-aot to generate fused forward+backward+SGD step.

  Example:
    (let [{:keys [train-fn A B]} (setup-matfac 8 6 3)]
      (dotimes [_ 1000]
        (train-fn A B X 8 6 3 0.01)))"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.blas :as blas]))

;; ================================================================
;; Matrix factorization loss: ||X - A @ B^T||^2 / (M*N)
;; ================================================================

(deftm matfac-loss [A :- (Array double) B :- (Array double)
                    X :- (Array double)
                    m :- Long n :- Long r :- Long] :- Double
  (let [;; Transpose B[N,r] -> B^T[r,N]
        Bt (nn/transpose-2d B n r)
        ;; pred = A[M,r] @ B^T[r,N] -> [M,N]
        pred (nn/matmul A Bt m r n)]
    ;; MSE loss
    (loss/mse-loss pred X (* m n))))

;; ================================================================
;; 2-layer MLP regression: simple function fitting
;; ================================================================

(deftm mlp-regression-loss [W1 :- (Array double) b1 :- (Array double)
                            W2 :- (Array double) b2 :- (Array double)
                            x :- (Array double) y :- (Array double)
                            in-dim :- Long hid-dim :- Long out-dim :- Long]
  :- Double
  (let [;; Layer 1: [1, in] @ W1^T[in, hid] + b1 -> [1, hid]
        h (nn/linear x W1 b1 1 in-dim hid-dim)
        ;; SiLU activation
        a (nn/silu h hid-dim)
        ;; Layer 2: [1, hid] @ W2^T[hid, out] + b2 -> [1, out]
        pred (nn/linear a W2 b2 1 hid-dim out-dim)]
    ;; MSE loss
    (loss/mse-loss pred y out-dim)))
