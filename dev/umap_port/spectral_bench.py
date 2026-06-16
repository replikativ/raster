"""Performance benchmark: umap's spectral_layout (scipy ARPACK eigsh / LOBPCG)
vs raster's matrix-free Lanczos spectral-init, on the IDENTICAL fuzzy graph.

We build umap's fuzzy_simplicial_set for several subset sizes, time umap's real
spectral_layout (the reference eigensolver we now compete with), and dump each
graph's symmetric edge list (head/tail/weights) so the raster driver solves the
exact same graph. Run from the umap tree:

    python3 dev/umap_port/spectral_bench.py
"""
import os, time
import numpy as np
import scipy.sparse, scipy.sparse.csgraph as csgraph
from sklearn.datasets import make_swiss_roll
from umap.umap_ import fuzzy_simplicial_set
from umap.spectral import spectral_layout

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, K, DIM = 42, 15, 2
SIZES = [1000, 2000, 5000]
REPS = 7

# Two regimes:
#   "mnist": 5000x50 digit subset (cleanly clustered -> ~10-component graph -> Tier C)
#   "swiss": swiss-roll manifold (continuous -> single connected component -> Tier A)
mnist = np.load(f"{OUT}/bench_X.npy").astype(np.float64).reshape(-1, 50)
DATASETS = {"mnist": mnist}
for n in SIZES:
    Xs, _ = make_swiss_roll(n_samples=n, random_state=SEED)
    DATASETS[f"swiss_{n}"] = Xs.astype(np.float64)
print(f"loaded mnist {mnist.shape}")

def median_ms(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))

def bench(name, X, n, metric):
    X = np.ascontiguousarray(X[:n])
    fgraph, _, _ = fuzzy_simplicial_set(
        X, K, random_state=np.random.RandomState(SEED), metric=metric)
    G = fgraph.tocoo()
    nc, _ = csgraph.connected_components(fgraph.tocsr())
    tag = f"{name}_{n}"
    np.save(f"{OUT}/sp_bench_head_{tag}.npy", G.row.astype(np.int32))
    np.save(f"{OUT}/sp_bench_tail_{tag}.npy", G.col.astype(np.int32))
    np.save(f"{OUT}/sp_bench_w_{tag}.npy", G.data.astype(np.float64))
    np.save(f"{OUT}/sp_bench_X_{tag}.npy", X.astype(np.float64))   # for raster Tier C centroids
    spectral_layout(None, fgraph, DIM, np.random.RandomState(SEED))   # warm up
    ms = median_ms(lambda: spectral_layout(None, fgraph, DIM, np.random.RandomState(SEED)), REPS)
    return (tag, G.nnz, nc, ms)

results = []
for n in SIZES:
    results.append(bench("mnist", mnist, n, "cosine"))                  # Tier C regime
    results.append(bench("swiss", DATASETS[f"swiss_{n}"], n, "euclidean"))  # Tier A regime
for tag, e, nc, ms in results:
    print(f"[{tag:11s}] edges={e:7d} components={nc:2d}  umap spectral_layout: {ms:8.2f} ms")

print("\nsummary (umap reference, scipy eigsh/LOBPCG):")
for tag, e, nc, ms in results:
    print(f"  [{tag:11s}]  {ms:8.2f} ms   ({e} edges, {nc} comp)")
print("\ngraphs+X dumped to sp_bench_{head,tail,w,X}_<tag>.npy for the raster driver")
