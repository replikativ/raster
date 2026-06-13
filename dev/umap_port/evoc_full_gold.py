"""Gold for the full EVoC pipeline (node_embedding_init=None config).

Dumps cosine kNN (knn_graph) for validating raster's knn-brute-cosine!, and the
full EVoC cluster labels for an end-to-end quality (ARI-vs-truth) comparison.

Run from the evoc tree:  PYTHONPATH=. python3 <repo>/dev/umap_port/evoc_full_gold.py
"""
import json, os
import numpy as np
from sklearn.datasets import make_blobs
from evoc.clustering import knn_graph, evoc_clusters

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, N, DIN, NN = 5, 300, 8, 15
X, y = make_blobs(n_samples=N, n_features=DIN, centers=5, random_state=SEED)
X = X.astype(np.float32)

inds, dists = knn_graph(X, n_neighbors=NN, random_state=np.random.RandomState(SEED))

res = evoc_clusters(X, n_neighbors=NN, node_embedding_init=None,
                    base_min_cluster_size=10, min_samples=5, n_epochs=50,
                    random_state=np.random.RandomState(SEED))
labels_e = np.asarray(res[0][0])     # first (base) layer cluster vector

def evec(a): return "[" + " ".join(repr(float(x)) for x in np.asarray(a).ravel()) + "]"
def eivec(a): return "[" + " ".join(str(int(x)) for x in np.asarray(a).ravel()) + "]"
with open(f"{OUT}/evoc_full_gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim-in {DIN} :n-neighbors {NN} ")
    f.write(":X " + evec(X.astype(np.float64)) + " ")
    f.write(":knn-idx " + eivec(inds) + " :knn-dst " + evec(dists) + " ")
    f.write(":labels " + eivec(labels_e) + "}")
np.save(f"{OUT}/evoc_full_labels.npy", y)
print(f"n={N} din={DIN} nn={NN} edges-knn={inds.shape}")
print("EVoC clusters:", len(set(labels_e[labels_e>=0])), "noise:", int((labels_e<0).sum()))
from sklearn.metrics import adjusted_rand_score as ari
print("EVoC ARI vs truth:", round(ari(labels_e, y), 4))
print("wrote evoc_full_gold.edn")
