"""Compare raster vs umap-learn embeddings: metrics table + side-by-side scatter.

Loads <ds>_{X,y,emb_umap,emb_raster}.npy (+ raster/umap meta) and computes, for
BOTH embeddings:
  - trustworthiness (sklearn, on a sample for large n) — local-structure preservation
  - 2D label-kNN accuracy — fraction of points whose k nearest embedding neighbors
    share its label (proxy for class separation)
plus runtimes. Renders <ds>_compare.png (raster | umap) and prints a table.

    python3 dev/umap_port/eval_plot.py mnist fashion
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.manifold import trustworthiness

OUT = "/tmp/umap_gold"
TAB10 = plt.get_cmap("tab10")
SEED = 42

def label_knn_acc(emb, y, k=10):
    tree = cKDTree(emb)
    _, nn = tree.query(emb, k=k+1)          # incl self
    return float((y[nn[:, 1:]] == y[:, None]).mean())

def trust(X, emb, n_neighbors=15, sample=5000):
    n = len(X)
    if n > sample:
        idx = np.random.RandomState(SEED).choice(n, sample, replace=False)
        X, emb = X[idx], emb[idx]
    return float(trustworthiness(X, emb, n_neighbors=n_neighbors))

def load(ds):
    X = np.load(f"{OUT}/{ds}_X.npy"); y = np.load(f"{OUT}/{ds}_y.npy")
    er = np.load(f"{OUT}/{ds}_emb_raster.npy"); eu = np.load(f"{OUT}/{ds}_emb_umap.npy")
    um = json.load(open(f"{OUT}/{ds}_meta.json"))
    rm = {}
    p = f"{OUT}/{ds}_raster_meta.edn"
    if os.path.exists(p):
        import re
        txt = open(p).read()
        for key in ("fit-secs", "knn-recall", "knn-secs"):
            m = re.search(rf":{key}\s+([0-9.]+)", txt)
            if m: rm[key] = float(m.group(1))
        m = re.search(r":init\s+:?(\w+)", txt); rm["init"] = m.group(1) if m else "?"
    return X, y, er, eu, um, rm

def scatter(ax, emb, y, title):
    ax.scatter(emb[:, 0], emb[:, 1], c=y % 10, cmap="tab10", s=2, alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

def main():
    datasets = sys.argv[1:] or ["fashion"]
    rows = []
    for ds in datasets:
        X, y, er, eu, um, rm = load(ds)
        n = len(X)
        m = {
            "ds": ds, "n": n,
            "raster_trust": trust(X, er), "umap_trust": trust(X, eu),
            "raster_lblacc": label_knn_acc(er, y), "umap_lblacc": label_knn_acc(eu, y),
            "raster_secs": rm.get("fit-secs", float("nan")), "umap_secs": um.get("umap_secs", float("nan")),
            "raster_recall": rm.get("knn-recall", float("nan")), "init": rm.get("init", "?"),
        }
        rows.append(m)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        scatter(axes[0], er, y, f"raster ({m['init']}) — trust {m['raster_trust']:.3f}  "
                                f"lbl-kNN {m['raster_lblacc']:.3f}  {m['raster_secs']:.1f}s")
        scatter(axes[1], eu, y, f"umap-learn — trust {m['umap_trust']:.3f}  "
                                f"lbl-kNN {m['umap_lblacc']:.3f}  {m['umap_secs']:.1f}s")
        fig.suptitle(f"{ds.upper()}  (n={n}, cosine)  raster vs umap-learn", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{ds}_compare.png", dpi=110); plt.close(fig)
        print(f"wrote {OUT}/{ds}_compare.png", flush=True)

    print("\n" + "=" * 100)
    hdr = f"{'dataset':<10}{'n':>7} {'init':>8} | {'trust(R)':>9}{'trust(U)':>9} | {'lblacc(R)':>10}{'lblacc(U)':>10} | {'recall(R)':>10} | {'sec(R)':>8}{'sec(U)':>8}"
    print(hdr); print("-" * 100)
    for m in rows:
        print(f"{m['ds']:<10}{m['n']:>7} {m['init']:>8} | {m['raster_trust']:>9.3f}{m['umap_trust']:>9.3f} | "
              f"{m['raster_lblacc']:>10.3f}{m['umap_lblacc']:>10.3f} | {m['raster_recall']:>10.4f} | "
              f"{m['raster_secs']:>8.1f}{m['umap_secs']:>8.1f}")
    print("=" * 100)
    json.dump(rows, open(f"{OUT}/compare_metrics.json", "w"), indent=2)

if __name__ == "__main__":
    main()
