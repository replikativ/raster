(ns raster.dl
  "Deep learning — public entry point.

  Provides layers, loss functions, optimizers, tensor operations,
  Einstein summation, and BLAS bindings.

  Usage:
    (require '[raster.dl :as dl])
    (dl/linear x W b batch-size in-dim out-dim)
    (dl/mse-loss pred target n)
    (dl/einsum \"ij,jk->ik\" A B)

  Sub-namespaces for advanced use:
    raster.dl.nn        — layers (matmul, linear, conv, norm, activation)
    raster.dl.loss      — loss functions
    raster.dl.tensor    — tensor creation and shape registry
    raster.dl.einsum    — Einstein summation, rearrange
    raster.dl.optim     — optimizers (SGD, Adam, AdamW, schedulers)
    raster.dl.train     — training utilities (data loader, train loop)
    raster.dl.attention — graph attention primitives
    raster.dl.diffusion — diffusion model utilities
    raster.dl.gsdm      — graphically structured diffusion model
    raster.dl.array-ops — low-level array primitives with AD support"
  (:require [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.tensor :as tensor]
            [raster.dl.einsum :as einsum]
            [raster.linalg.blas :as blas]
            [raster.dl.optim :as optim]
            [raster.dl.train :as train]
            [raster.support :refer [import-vars]]))

;; ================================================================
;; Layers (raster.dl.nn)
;; ================================================================

(import-vars raster.dl.nn
             matmul linear silu gelu sigmoid tanh-act leaky-relu
             layer-norm group-norm batch-norm
             conv1d conv2d maxpool2d
             embedding dropout
             transpose-2d)

;; ================================================================
;; Loss functions (raster.dl.loss)
;; ================================================================

(import-vars raster.dl.loss
             mse-loss cross-entropy-loss huber-loss l1-loss)

;; ================================================================
;; Tensors (raster.dl.tensor)
;; ================================================================

(import-vars raster.dl.tensor
             tensor tshape trank tsize tidx strides
             zeros ones full randn rand-uniform arange linspace
             reshape flatten-tensor
             tensor-add tensor-scale tensor-eq?)

;; ================================================================
;; Einstein summation (raster.dl.einsum)
;; ================================================================

(import-vars raster.dl.einsum
             einsum rearrange)

;; ================================================================
;; Optimizers (raster.dl.optim)
;; ================================================================

(import-vars raster.dl.optim
             sgd-step! adam-step! adamw-step!
             clip-grad-norm! ema-update!
             cosine-lr warmup-cosine-lr step-lr linear-warmup-lr
             make-adam-state adam-update!)

;; ================================================================
;; Training (raster.dl.train)
;; ================================================================

(import-vars raster.dl.train
             data-loader collate-doubles collate-longs train-epoch!)
