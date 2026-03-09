#!/usr/bin/env python3
"""JAX MNIST benchmarks for fair comparison with Raster.

Benchmark 1: 784-128-10 MLP, single-sample SGD, Float64
Benchmark 2: 784-128-10 MLP, single-sample SGD, Float32
Benchmark 3: LeNet-5, single-sample SGD, Float64

Run: OPENBLAS_NUM_THREADS=1 python3 bench/jax_mnist_bench.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import gzip
import struct
import time
import numpy as np

jax.config.update("jax_platform_name", "cpu")

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>4I', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        return data.astype(np.float64) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>2I', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot(labels, num_classes=10):
    return np.eye(num_classes, dtype=np.float64)[labels]

# Load data
print("Loading MNIST...")
train_x = load_mnist_images("data/mnist/train-images-idx3-ubyte.gz")
train_y_int = load_mnist_labels("data/mnist/train-labels-idx1-ubyte.gz")
train_y = one_hot(train_y_int)
test_x = load_mnist_images("data/mnist/t10k-images-idx3-ubyte.gz")
test_y_int = load_mnist_labels("data/mnist/t10k-labels-idx1-ubyte.gz")
n_train = len(train_x)
n_test = len(test_x)
print(f"Train: {n_train}, Test: {n_test}")

# ================================================================
# MLP helpers
# ================================================================

def init_mlp(rng, hidden=128, dtype=jnp.float64):
    k1, k2 = jax.random.split(rng)
    scale1 = np.sqrt(6.0 / (784 + hidden))
    scale2 = np.sqrt(6.0 / (hidden + 10))
    W1 = jax.random.uniform(k1, (784, hidden), minval=-scale1, maxval=scale1, dtype=dtype)
    b1 = jnp.zeros(hidden, dtype=dtype)
    W2 = jax.random.uniform(k2, (hidden, 10), minval=-scale2, maxval=scale2, dtype=dtype)
    b2 = jnp.zeros(10, dtype=dtype)
    return (W1, b1, W2, b2)

def mlp_loss(params, x, y):
    W1, b1, W2, b2 = params
    h = jnp.maximum(x @ W1 + b1, 0.0)  # ReLU
    logits = h @ W2 + b2
    probs = jax.nn.softmax(logits)
    return -jnp.sum(y * jnp.log(probs + 1e-10))

@jit
def mlp_step(params, x, y, lr):
    loss, grads = value_and_grad(mlp_loss)(params, x, y)
    params = tuple(p - lr * g for p, g in zip(params, grads))
    return params, loss

def mlp_predict(params, x):
    W1, b1, W2, b2 = params
    h = jnp.maximum(x @ W1 + b1, 0.0)
    logits = h @ W2 + b2
    return jnp.argmax(logits, axis=-1)

# ================================================================
# Benchmark 1: MLP Float64
# ================================================================

print(f"\n=== JAX MNIST MLP Benchmark (float64, CPU) ===")
print(f"JAX {jax.__version__}, devices: {jax.devices()}")
print("Architecture: 784-128-10, SGD lr=0.01")

params = init_mlp(jax.random.PRNGKey(42), dtype=jnp.float64)
train_x64 = jnp.array(train_x, dtype=jnp.float64)
train_y64 = jnp.array(train_y, dtype=jnp.float64)
test_x64 = jnp.array(test_x, dtype=jnp.float64)

# Warmup
print("Warming up JIT...", end=" ", flush=True)
t0 = time.time()
for i in range(1000):
    params, _ = mlp_step(params, train_x64[i % n_train], train_y64[i % n_train], 0.01)
_ = jax.block_until_ready(params[0])
print(f"done ({time.time()-t0:.1f}s)")

# Re-init after warmup
params = init_mlp(jax.random.PRNGKey(42), dtype=jnp.float64)

for epoch in range(3):
    indices = np.random.permutation(n_train)
    t0 = time.time()
    total_loss = 0.0
    for i in indices:
        params, loss = mlp_step(params, train_x64[i], train_y64[i], 0.01)
        total_loss += float(loss)
    _ = jax.block_until_ready(params[0])
    dt = time.time() - t0
    avg_loss = total_loss / n_train
    preds = np.array(jnp.argmax(jnp.maximum(test_x64 @ params[0] + params[1], 0.0) @ params[2] + params[3], axis=-1))
    acc = np.mean(preds == test_y_int) * 100
    print(f"Epoch {epoch+1}/3  loss={avg_loss:.4f}  acc={acc:.2f}%  time={dt:.1f}s  ({dt/n_train*1e6:.0f} µs/step)")

# ================================================================
# Benchmark 2: MLP Float32
# ================================================================

print(f"\n=== JAX MNIST MLP Benchmark (float32, CPU) ===")
print("Architecture: 784-128-10, SGD lr=0.01")

params32 = init_mlp(jax.random.PRNGKey(42), dtype=jnp.float32)
train_x32 = jnp.array(train_x, dtype=jnp.float32)
train_y32 = jnp.array(train_y, dtype=jnp.float32)
test_x32 = jnp.array(test_x, dtype=jnp.float32)

@jit
def mlp_step32(params, x, y, lr):
    loss, grads = value_and_grad(mlp_loss)(params, x, y)
    params = tuple(p - lr * g for p, g in zip(params, grads))
    return params, loss

# Warmup
print("Warming up JIT...", end=" ", flush=True)
t0 = time.time()
for i in range(1000):
    params32, _ = mlp_step32(params32, train_x32[i % n_train], train_y32[i % n_train], 0.01)
_ = jax.block_until_ready(params32[0])
print(f"done ({time.time()-t0:.1f}s)")

params32 = init_mlp(jax.random.PRNGKey(42), dtype=jnp.float32)

for epoch in range(3):
    indices = np.random.permutation(n_train)
    t0 = time.time()
    total_loss = 0.0
    for i in indices:
        params32, loss = mlp_step32(params32, train_x32[i], train_y32[i], 0.01)
        total_loss += float(loss)
    _ = jax.block_until_ready(params32[0])
    dt = time.time() - t0
    avg_loss = total_loss / n_train
    preds = np.array(jnp.argmax(jnp.maximum(test_x32 @ params32[0] + params32[1], 0.0) @ params32[2] + params32[3], axis=-1))
    acc = np.mean(preds == test_y_int) * 100
    print(f"Epoch {epoch+1}/3  loss={avg_loss:.4f}  acc={acc:.2f}%  time={dt:.1f}s")

# ================================================================
# Benchmark 3: LeNet-5 Float64
# ================================================================

print(f"\n=== JAX LeNet-5 MNIST Benchmark (float64, CPU) ===")
print("Architecture: Conv(1→6,5x5)→Pool→Conv(6→16,5x5)→Pool→FC(256→120)→FC(120→10)")
print("Single-sample SGD lr=0.01")

def init_lenet(rng, dtype=jnp.float64):
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    # Kaiming init
    conv1_W = jax.random.normal(k1, (6, 1, 5, 5), dtype=dtype) * jnp.sqrt(2.0 / 25.0).astype(dtype)
    conv1_b = jnp.zeros(6, dtype=dtype)
    conv2_W = jax.random.normal(k2, (16, 6, 5, 5), dtype=dtype) * jnp.sqrt(2.0 / 150.0).astype(dtype)
    conv2_b = jnp.zeros(16, dtype=dtype)
    fc1_W = jax.random.normal(k3, (120, 256), dtype=dtype) * jnp.sqrt(2.0 / 256.0).astype(dtype)
    fc1_b = jnp.zeros(120, dtype=dtype)
    fc2_W = jax.random.normal(k4, (10, 120), dtype=dtype) * jnp.sqrt(2.0 / 120.0).astype(dtype)
    fc2_b = jnp.zeros(10, dtype=dtype)
    return (conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b)

def lenet_forward(params, x):
    conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc1_b, fc2_W, fc2_b = params
    # Reshape flat input to [1, 1, 28, 28]
    x = x.reshape(1, 1, 28, 28)
    # Conv1: [1,1,28,28] → [1,6,24,24]
    # JAX conv uses [N,C,H,W] with dimension_numbers
    c1 = jax.lax.conv(x, conv1_W, window_strides=(1,1), padding='VALID') + conv1_b.reshape(1,-1,1,1)
    r1 = jnp.maximum(c1, 0.0)
    # MaxPool 2x2 → [1,6,12,12]
    p1 = -jax.lax.reduce_window(-r1, jnp.inf, jax.lax.min, (1,1,2,2), (1,1,2,2), 'VALID')
    # Conv2: [1,6,12,12] → [1,16,8,8]
    c2 = jax.lax.conv(p1, conv2_W, window_strides=(1,1), padding='VALID') + conv2_b.reshape(1,-1,1,1)
    r2 = jnp.maximum(c2, 0.0)
    # MaxPool 2x2 → [1,16,4,4]
    p2 = -jax.lax.reduce_window(-r2, jnp.inf, jax.lax.min, (1,1,2,2), (1,1,2,2), 'VALID')
    # Flatten → [256]
    flat = p2.reshape(-1)
    # FC1: 256→120
    f1 = fc1_W @ flat + fc1_b
    a1 = jnp.maximum(f1, 0.0)
    # FC2: 120→10
    f2 = fc2_W @ a1 + fc2_b
    return f2

def lenet_loss(params, x, y):
    logits = lenet_forward(params, x)
    probs = jax.nn.softmax(logits)
    return -jnp.sum(y * jnp.log(probs + 1e-10))

@jit
def lenet_step(params, x, y, lr):
    loss, grads = value_and_grad(lenet_loss)(params, x, y)
    params = tuple(p - lr * g for p, g in zip(params, grads))
    return params, loss

lenet_params = init_lenet(jax.random.PRNGKey(42))

# Warmup
print("Warming up JIT...", end=" ", flush=True)
t0 = time.time()
for i in range(200):
    lenet_params, _ = lenet_step(lenet_params, train_x64[i % n_train], train_y64[i % n_train], 0.01)
_ = jax.block_until_ready(lenet_params[0])
print(f"done ({time.time()-t0:.1f}s)")

# Per-step timing (1000 steps, warmed up)
lenet_params_bench = init_lenet(jax.random.PRNGKey(42))
n_bench = 1000
t0 = time.time()
for i in range(n_bench):
    lenet_params_bench, _ = lenet_step(lenet_params_bench, train_x64[i % n_train], train_y64[i % n_train], 0.01)
_ = jax.block_until_ready(lenet_params_bench[0])
dt_bench = time.time() - t0
print(f"Per-step: {dt_bench/n_bench*1e6:.0f} µs/step ({n_bench} steps)")

lenet_params = init_lenet(jax.random.PRNGKey(42))

for epoch in range(3):
    indices = np.random.permutation(n_train)
    t0 = time.time()
    total_loss = 0.0
    for i in indices:
        lenet_params, loss = lenet_step(lenet_params, train_x64[i], train_y64[i], 0.01)
        total_loss += float(loss)
    _ = jax.block_until_ready(lenet_params[0])
    dt = time.time() - t0
    avg_loss = total_loss / n_train
    # Quick eval
    correct = 0
    for i in range(n_test):
        logits = lenet_forward(lenet_params, test_x64[i])
        if int(jnp.argmax(logits)) == test_y_int[i]:
            correct += 1
    acc = correct / n_test * 100
    print(f"Epoch {epoch+1}/3  loss={avg_loss:.4f}  acc={acc:.2f}%  time={dt:.1f}s  ({dt/n_train*1e6:.0f} µs/step)")

print("\nDone.")
