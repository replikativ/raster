"""Generate datasets + umap-learn reference for the raster-vs-reference comparison.

For each dataset we dump (ALL arrays np.ascontiguousarray, C-order — raster's
readers honor fortran_order now, but C-order keeps everything simple and fast):

    <ds>_X.npy          float64  (n, dim)   raw features
    <ds>_y.npy          int32    (n,)       labels
    <ds>_emb_umap.npy   float64  (n, 2)     reference umap-learn embedding
    <ds>_knnq.npy       int32    (q,)       sample query indices
    <ds>_knne.npy       int32    (q, K)     exact cosine kNN of the queries (incl self)
    <ds>_meta.json                          {umap_secs, n, dim, k, metric}

Usage:
    python3 dev/umap_port/gen_datasets.py mnist 70000
    python3 dev/umap_port/gen_datasets.py fashion 70000
    python3 dev/umap_port/gen_datasets.py mnist 7000        # quick iteration
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize
import umap

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, K = 42, 15
QSAMPLE = 3000   # exact-kNN query sample size (for recall / trustworthiness refs)

OPENML = {"mnist": ("mnist_784", 1), "fashion": ("Fashion-MNIST", 1)}

def fetch(name):
    key, ver = OPENML[name]
    d = fetch_openml(key, version=ver, as_frame=False)
    return d.data.astype(np.float64), d.target.astype(np.int32)

def exact_cosine_knn(X, qidx, k):
    """Exact cosine kNN of the query rows against all rows, via chunked matmul."""
    Xn = normalize(X)
    Q = Xn[qidx]                       # (q, dim)
    out = np.empty((len(qidx), k), np.int32)
    step = 512
    for s in range(0, len(qidx), step):
        sims = Q[s:s+step] @ Xn.T      # (step, n) cosine sims
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        # order each row's top-k by similarity desc
        rows = np.arange(idx.shape[0])[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        out[s:s+step] = idx[rows, order]
    return out

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 70000
    X_all, y_all = fetch(name)
    rng = np.random.RandomState(SEED)
    if N < len(X_all):
        sel = rng.choice(len(X_all), N, replace=False)
        X, y = X_all[sel], y_all[sel]
    else:
        X, y = X_all, y_all
        N = len(X)
    X = np.ascontiguousarray(X.astype(np.float64))
    y = np.ascontiguousarray(y.astype(np.int32))
    dim = X.shape[1]
    print(f"[{name}] X {X.shape} y {y.shape} classes {len(set(y.tolist()))}", flush=True)

    np.save(f"{OUT}/{name}_X.npy", X)
    np.save(f"{OUT}/{name}_y.npy", y)

    print(f"[{name}] umap-learn (cosine) fitting ...", flush=True)
    t0 = time.perf_counter()
    emb = umap.UMAP(n_neighbors=K, min_dist=0.1, metric="cosine",
                    random_state=SEED).fit_transform(X)
    secs = time.perf_counter() - t0
    np.save(f"{OUT}/{name}_emb_umap.npy", np.ascontiguousarray(emb.astype(np.float64)))
    print(f"[{name}] umap fit {secs:.1f}s", flush=True)

    qidx = np.ascontiguousarray(rng.choice(N, min(QSAMPLE, N), replace=False).astype(np.int32))
    print(f"[{name}] exact cosine kNN for {len(qidx)} query points ...", flush=True)
    knne = np.ascontiguousarray(exact_cosine_knn(X, qidx, K))
    np.save(f"{OUT}/{name}_knnq.npy", qidx)
    np.save(f"{OUT}/{name}_knne.npy", knne)

    json.dump({"umap_secs": secs, "n": int(N), "dim": int(dim), "k": K, "metric": "cosine"},
              open(f"{OUT}/{name}_meta.json", "w"))
    print(f"[{name}] done -> {OUT}/{name}_{{X,y,emb_umap,knnq,knne}}.npy + meta.json", flush=True)

if __name__ == "__main__":
    main()
