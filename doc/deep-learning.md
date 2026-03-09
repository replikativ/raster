# Deep Learning

The `raster.dl` namespace provides neural network layers, loss functions,
optimizers, and a compiled training pipeline. All layers are `deftm` functions
operating on flat arrays with explicit shape parameters, so the compiler sees
through layer boundaries and can fuse operations end-to-end.

## Layers (`raster.dl.nn`)

| Layer | Function |
|-------|----------|
| Dense | `linear` |
| Convolution | `conv1d`, `conv2d` |
| Pooling | `maxpool2d` |
| Normalization | `layer-norm`, `group-norm`, `batch-norm` |
| Activations | `relu`, `silu`, `gelu`, `sigmoid`, `tanh-act`, `leaky-relu` |
| Other | `embedding`, `dropout` |

Matmul is backed by OpenBLAS via Panama FFI (`cblas_dgemv`/`cblas_sgemv`).

## Attention (`raster.dl.attention`)

Scaled dot-product attention, multi-head self-attention, and graph attention
with scatter-based message passing.

## Loss Functions (`raster.dl.loss`)

MSE, cross-entropy (numerically stable logit form), Huber, and L1 loss — all
with rrules for reverse AD.

## Optimizers (`raster.dl.optim`)

SGD, Adam, AdamW with in-place updates. Gradient clipping, EMA, and learning
rate schedulers (cosine, warmup-cosine, step, linear warmup).

## Compiled Training

For production performance, `compile-aot` compiles the entire training step —
forward pass, backward pass (reverse AD), and optimizer update — into one
fused JVM method:

```clojure
(def fast-step (pipeline/compile-aot #'train-step!))

(dotimes [i n-epochs]
  (fast-step W1 b1 W2 b2 x-batch y-batch lr))
```

The compiler performs buffer fusion (reusing dead arrays), SOAC fusion
(merging parallel operations), and SIMD vectorization — all automatically.

## Einsum and Tensor Ops

Einstein summation notation and einops-style rearrangement on flat arrays:

```clojure
(einsum "ij,jk->ik" A B)        ;; matrix multiply
(einsum "bij,bjk->bik" A B)     ;; batch matmul
(rearrange A "b c h w -> b (c h) w")  ;; merge dims
```

## Reference Models

- [LeNet-5](../examples/raster/dl/lenet.clj) — compiled CNN on MNIST
- [GPT-2](../examples/raster/dl/gpt2.clj) — transformer language model
- [BERT](../examples/raster/dl/bert.clj) — masked language model
- [Matrix Factorization](../examples/raster/dl/matfac.clj) — collaborative filtering
