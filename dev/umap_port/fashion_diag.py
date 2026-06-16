"""Diagnostic: dump umap's fuzzy graph + exact cosine kNN for the Fashion-MNIST
subset, so we can (a) compare raster's kNN/graph to umap's, and (b) feed raster's
SGD the exact umap graph to isolate where the quality gap lives."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import fuzzy_simplicial_set

OUT = "/tmp/umap_gold"
SEED, K = 42, 15
X = np.load(f"{OUT}/fashion_X.npy").astype(np.float32)
n = X.shape[0]
print(f"X {X.shape}", flush=True)

# umap's fuzzy simplicial set (cosine), the graph simplicial_set_embedding uses
fgraph, sig, rho = fuzzy_simplicial_set(
    X, K, random_state=np.random.RandomState(SEED), metric="cosine")
G = fgraph.tocoo()
np.save(f"{OUT}/fashion_g_head.npy", G.row.astype(np.int32))
np.save(f"{OUT}/fashion_g_tail.npy", G.col.astype(np.int32))
np.save(f"{OUT}/fashion_g_w.npy", G.data.astype(np.float64))
print(f"umap fuzzy graph: {G.nnz} edges", flush=True)

# exact cosine kNN (for recall comparison vs raster's NN-descent)
nn = NearestNeighbors(n_neighbors=K, metric="cosine").fit(X)
_, idx = nn.kneighbors(X)
np.save(f"{OUT}/fashion_knn_exact.npy", idx.astype(np.int32))
print(f"exact cosine kNN dumped ({idx.shape})")
print("wrote fashion_g_{head,tail,w}.npy + fashion_knn_exact.npy")
