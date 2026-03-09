"""Benchmark scikit-learn PCA for comparison with Raster."""
import numpy as np
from sklearn.decomposition import PCA
import time

def bench(m, n, k, label):
    rng = np.random.RandomState(42)
    X = rng.randn(m, n)

    # Warmup
    PCA(n_components=k, svd_solver='full').fit(X)

    iters = max(3, min(20, int(500 / max(m, n))))

    # Full SVD
    t0 = time.perf_counter()
    for _ in range(iters):
        PCA(n_components=k, svd_solver='full').fit(X)
    t_full = (time.perf_counter() - t0) / iters * 1000

    # Covariance eigh (sklearn 1.5+) or skip
    try:
        PCA(n_components=k, svd_solver='covariance_eigh').fit(X)
        t1 = time.perf_counter()
        for _ in range(iters):
            PCA(n_components=k, svd_solver='covariance_eigh').fit(X)
        t_eigh = (time.perf_counter() - t1) / iters * 1000
    except Exception:
        t_eigh = float('nan')

    # Randomized
    t2 = time.perf_counter()
    for _ in range(iters):
        PCA(n_components=k, svd_solver='randomized', n_oversamples=10, iterated_power=4).fit(X)
    t_rand = (time.perf_counter() - t2) / iters * 1000

    print(f"{label} [{m}x{n}, k={k}]:  full={t_full:.1f}ms  eigh={t_eigh:.1f}ms  rand={t_rand:.1f}ms")

if __name__ == '__main__':
    bench(100, 50, 10, "Small")
    bench(500, 100, 10, "Medium-1")
    bench(1000, 500, 10, "Medium-2")
    bench(5000, 100, 10, "Tall")
    bench(1000, 500, 50, "Med-k50")
    bench(10000, 1000, 10, "Large")
    bench(10000, 100, 10, "Large-tall")
