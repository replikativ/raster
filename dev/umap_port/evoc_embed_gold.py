"""Gold for EVoC node_embedding (the SGD on the fuzzy graph).

Drives node_embedding_epoch manually with a FIXED init + FIXED rng_vals so raster
can reproduce with identical inputs (the negative index is deterministic uint32
arithmetic, so only float32-vs-float64 differs). Dumps an attractive-only run
(negatives disabled via huge epn) for a tight deterministic gate, plus a full run.

Run from the evoc tree:  PYTHONPATH=. python3 <repo>/dev/umap_port/evoc_embed_gold.py
"""
import json, os
import numpy as np
from sklearn.datasets import make_blobs
from evoc.clustering import knn_graph, neighbor_graph_matrix
from evoc.node_embedding import node_embedding_epoch, make_epochs_per_sample

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, N, DIN, NN, DIM, NE = 11, 200, 8, 15, 4, 30
NOISE, INIT_ALPHA = 0.5, 0.1

rng = np.random.RandomState(SEED)
X, y = make_blobs(n_samples=N, n_features=DIN, centers=4, random_state=SEED)
X = X.astype(np.float32)
nn_inds, nn_dists = knn_graph(X, n_neighbors=NN, random_state=np.random.RandomState(SEED))
graph = neighbor_graph_matrix(NN, nn_inds, nn_dists, True).tocoo()
head = graph.row.astype(np.uint32); tail = graph.col.astype(np.uint32)
eps = make_epochs_per_sample(graph.data, NE).astype(np.float32)

init = rng.normal(scale=0.25, size=(N, DIM)).astype(np.float32, order="C")
rng_vals = rng.randint(np.iinfo(np.int32).max, size=NE).astype(np.int64)

def run(attractive_only):
    emb = init.copy()
    epn = (np.full_like(eps, 1e18) if attractive_only else eps / 1.0)   # neg_rate=1
    eonsn = epn.copy(); eons = eps.copy(); alpha = np.float32(INIT_ALPHA)
    for n in range(NE):
        node_embedding_epoch(emb, head, tail, np.uint32(N), eps,
                             np.uint32(rng_vals[n]), np.uint8(DIM), np.float32(alpha),
                             epn, eonsn, eons, np.uint8(n), np.float32(NOISE))
        alpha = np.float32(INIT_ALPHA * (1.0 - (float(n) / float(NE))))
    return emb

emb_attr = run(True)
emb_full = run(False)

def evec(a): return "[" + " ".join(repr(float(x)) for x in np.asarray(a).ravel()) + "]"
def eivec(a): return "[" + " ".join(str(int(x)) for x in np.asarray(a).ravel()) + "]"
with open(f"{OUT}/evoc_embed_gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim {DIM} :n-epochs {NE} :noise {NOISE} :init-alpha {INIT_ALPHA} ")
    f.write(":head " + eivec(head) + " :tail " + eivec(tail) + " :eps " + evec(eps) + " ")
    f.write(":rng-vals " + eivec(rng_vals) + " ")
    f.write(":init " + evec(init.astype(np.float64)) + " ")
    f.write(":emb-attr " + evec(emb_attr.astype(np.float64)) + " ")
    f.write(":emb-full " + evec(emb_full.astype(np.float64)) + "}")
np.save(f"{OUT}/evoc_embed_X.npy", X.astype(np.float64))
np.save(f"{OUT}/evoc_embed_labels.npy", y)
print(f"n={N} dim={DIM} epochs={NE} edges={head.shape[0]} noise={NOISE}")
print("emb_full range:", round(float(emb_full.min()),3), round(float(emb_full.max()),3))
print("wrote evoc_embed_gold.edn")
