#!/usr/bin/env python3
"""Dump XLA HLO and native assembly for the MLP 784-128-10 train step.

Produces:
  <outdir>/xla_hlo_optimized.txt   — XLA HLO after optimization
  <outdir>/xla_hlo_unoptimized.txt — XLA HLO before optimization
  <outdir>/xla_timing.txt          — micro-benchmark timing (µs/step)
  <outdir>/xla_fusion_summary.txt  — kernel count and fusion summary

Usage:
  OPENBLAS_NUM_THREADS=1 python3 bench/comparison/jax_mlp_assembly.py [outdir]

Default outdir: /tmp/raster_vs_xla/
"""

import os, sys, time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import numpy as np

outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/raster_vs_xla"
os.makedirs(outdir, exist_ok=True)

# ================================================================
# MLP 784-128-10 — identical to jax_mnist_bench.py
# ================================================================

def mlp_loss(params, x, y):
    W1, b1, W2, b2 = params
    h = jnp.maximum(x @ W1 + b1, 0.0)  # dense1 + relu
    logits = h @ W2 + b2                 # dense2
    probs = jax.nn.softmax(logits)
    return -jnp.sum(y * jnp.log(jnp.maximum(probs, 1e-15)))

@jit
def mlp_step(params, x, y, lr):
    loss, grads = value_and_grad(mlp_loss)(params, x, y)
    params = tuple(p - lr * g for p, g in zip(params, grads))
    return params, loss

# ================================================================
# Initialize
# ================================================================

rng = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(rng)
W1 = jax.random.normal(k1, (784, 128), dtype=jnp.float64) * 0.01
b1 = jnp.zeros(128, dtype=jnp.float64)
W2 = jax.random.normal(k2, (128, 10), dtype=jnp.float64) * 0.01
b2 = jnp.zeros(10, dtype=jnp.float64)
params = (W1, b1, W2, b2)

x = jnp.zeros(784, dtype=jnp.float64)
y = jnp.zeros(10, dtype=jnp.float64).at[3].set(1.0)

# ================================================================
# 1. Dump HLO
# ================================================================

print("Lowering and compiling...")
lowered = jax.jit(mlp_step).lower(params, x, y, 0.01)

# Unoptimized HLO
with open(os.path.join(outdir, "xla_hlo_unoptimized.txt"), "w") as f:
    f.write(lowered.as_text())
print(f"  wrote {outdir}/xla_hlo_unoptimized.txt")

# Compiled (optimized) HLO
compiled = lowered.compile()
with open(os.path.join(outdir, "xla_hlo_optimized.txt"), "w") as f:
    f.write(compiled.as_text())
print(f"  wrote {outdir}/xla_hlo_optimized.txt")

# ================================================================
# 2. XLA dump (includes native assembly if XLA_FLAGS set)
# ================================================================

# Attempt to get cost analysis
try:
    cost = compiled.cost_analysis()
    with open(os.path.join(outdir, "xla_cost_analysis.txt"), "w") as f:
        for item in cost:
            f.write(str(item) + "\n")
    print(f"  wrote {outdir}/xla_cost_analysis.txt")
except Exception as e:
    print(f"  cost_analysis not available: {e}")

# ================================================================
# 3. Fusion summary — count kernels in optimized HLO
# ================================================================

hlo_text = compiled.as_text()
fusions = [l.strip() for l in hlo_text.split('\n')
           if 'fusion' in l.lower() and l.strip().startswith('%')]
computations = [l.strip() for l in hlo_text.split('\n')
                if l.strip().startswith('ENTRY') or l.strip().startswith('fused_computation')]

with open(os.path.join(outdir, "xla_fusion_summary.txt"), "w") as f:
    f.write(f"Total fusion instructions: {len(fusions)}\n")
    f.write(f"Total fused_computation blocks: {len(computations)}\n\n")
    f.write("Fusion instructions:\n")
    for line in fusions[:50]:
        f.write(f"  {line}\n")
    f.write("\nComputation blocks:\n")
    for line in computations[:30]:
        f.write(f"  {line}\n")
print(f"  wrote {outdir}/xla_fusion_summary.txt")

# ================================================================
# 4. Micro-benchmark (matches Raster bench protocol)
# ================================================================

print("Benchmarking...")

# Warmup
for _ in range(5000):
    params, _ = mlp_step(params, x, y, 0.01)
jax.block_until_ready(params[0])

# Re-init for clean benchmark
params = (W1, b1, W2, b2)

# 200 batches of 200 steps each (matches Raster protocol)
times = []
for i in range(200):
    t0 = time.perf_counter_ns()
    for _ in range(200):
        params, _ = mlp_step(params, x, y, 0.01)
    jax.block_until_ready(params[0])
    dt = time.perf_counter_ns() - t0
    times.append(dt / 200_000)  # µs per step

times.sort()
median = times[100]
p10 = times[20]
p90 = times[180]
minimum = times[0]

with open(os.path.join(outdir, "xla_timing.txt"), "w") as f:
    f.write(f"JAX {jax.__version__} CPU, float64, MLP 784-128-10, SGD lr=0.01\n")
    f.write(f"Protocol: 5000 warmup, 200 batches of 200 steps\n\n")
    f.write(f"median:  {median:.1f} µs\n")
    f.write(f"p10:     {p10:.1f} µs\n")
    f.write(f"p90:     {p90:.1f} µs\n")
    f.write(f"min:     {minimum:.1f} µs\n")
print(f"  wrote {outdir}/xla_timing.txt")
print(f"\n  JAX MLP f64: median={median:.1f} µs, min={minimum:.1f} µs")

# ================================================================
# 5. For full native assembly, re-run with:
#    XLA_FLAGS="--xla_dump_to=/tmp/raster_vs_xla/xla_native --xla_dump_hlo_as_text" \
#    OPENBLAS_NUM_THREADS=1 python3 bench/comparison/jax_mlp_assembly.py
# ================================================================

print(f"""
For native x86 assembly, re-run with:
  XLA_FLAGS="--xla_dump_to={outdir}/xla_native --xla_dump_hlo_as_text" \\
  OPENBLAS_NUM_THREADS=1 python3 {sys.argv[0]}

Output will include .s files with the generated x86 code.
""")
