"""Fashion-MNIST: umap reference embedding + dump the identical subset for raster.

Fetches Fashion-MNIST (openml), takes a class-balanced subset, runs umap with the
SAME metric raster uses (cosine) on raw pixels, saves the embedding + timing, and
dumps X/y so the raster fit driver embeds the identical data. The Clojure side then
plots both side by side. Run from the umap tree:

    python3 dev/umap_port/fashion_compare.py [N]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import fetch_openml
import umap

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED, K = 42, 15

print(f"fetching Fashion-MNIST ...", flush=True)
fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
X_all = fm.data.astype(np.float32)
y_all = fm.target.astype(np.int32)
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X_all), N, replace=False)
X = np.ascontiguousarray(X_all[idx]); y = np.ascontiguousarray(y_all[idx])
print(f"subset X {X.shape} y {y.shape} classes {sorted(set(y.tolist()))}", flush=True)

np.save(f"{OUT}/fashion_X.npy", X.astype(np.float64))
np.save(f"{OUT}/fashion_y.npy", y)

t0 = time.perf_counter()
emb = umap.UMAP(n_neighbors=K, min_dist=0.1, metric="cosine",
                random_state=SEED).fit_transform(X)
secs = time.perf_counter() - t0
np.save(f"{OUT}/fashion_emb_umap.npy", emb.astype(np.float64))


def nn_agree(emb, y):
    # fraction of points whose nearest 2D neighbour shares its label
    from scipy.spatial import cKDTree
    d, i = cKDTree(emb).query(emb, k=2)
    return float((y[i[:, 1]] == y).mean())


print(f"umap cosine fit: {secs:.1f}s  2D-NN-agree {nn_agree(emb, y):.3f}", flush=True)
print(f"wrote fashion_X/y.npy + fashion_emb_umap.npy (N={N})")
