"""Connected-graph gold case for validating raster spectral init.

Blob graphs are disconnected (-> umap multi_component_layout); a swiss-roll's
fuzzy graph is a single connected component, exercising the core normalized-
Laplacian smallest-eigenvector path. Reference eigenpairs computed with dense
LAPACK (np.linalg.eigh) — exact, same routine raster's eigh wraps.

Run from the umap tree:  python3 dev/umap_port/spectral_gold.py
"""
import json, os
import numpy as np
import scipy.sparse, scipy.sparse.csgraph as csgraph
from sklearn.datasets import make_swiss_roll
from umap.umap_ import fuzzy_simplicial_set

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, N, K, DIM = 7, 400, 15, 2
X, color = make_swiss_roll(n_samples=N, random_state=SEED)
X = X.astype(np.float32)

fgraph, sig, rho = fuzzy_simplicial_set(
    X, K, random_state=np.random.RandomState(SEED), metric="euclidean")
G = fgraph.tocoo()
A = G.tocsr()
nc, labels = csgraph.connected_components(A)
assert nc == 1, f"need connected graph, got {nc} components"

# dense normalized Laplacian L = I - D^-1/2 A D^-1/2
Ad = A.toarray().astype(np.float64)
deg = Ad.sum(axis=0)
isd = 1.0 / np.sqrt(deg)
L = np.eye(N) - (isd[:, None] * Ad * isd[None, :])
evals, evecs = np.linalg.eigh(L)        # ascending; evecs[:,i] = i-th eigenvector
# smallest dim+1 (index 0 is the trivial ~0 eigenvalue)
sm_evals = evals[:DIM + 1]
sm_evecs = evecs[:, :DIM + 1]            # columns

def evec(a): return "[" + " ".join(repr(float(x)) for x in np.asarray(a).ravel()) + "]"
def eivec(a): return "[" + " ".join(str(int(x)) for x in np.asarray(a).ravel()) + "]"

np.save(f"{OUT}/sp_X.npy", X.astype(np.float64))
np.save(f"{OUT}/sp_evals.npy", sm_evals)
np.save(f"{OUT}/sp_evecs.npy", sm_evecs)
np.save(f"{OUT}/sp_color.npy", color)

with open(f"{OUT}/spectral_gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim {DIM} :k {K} :dim-in {X.shape[1]} ")
    f.write(":X " + evec(X.astype(np.float64)) + " ")
    f.write(":g-row " + eivec(G.row) + " :g-col " + eivec(G.col) + " :g-val " + evec(G.data) + " ")
    f.write(":evals " + evec(sm_evals) + " ")
    f.write(":evecs-cols " + evec(sm_evecs.T) + "}")    # row c = c-th eigenvector

print(f"n={N} edges={G.nnz} components={nc}")
print("smallest dim+1 eigenvalues:", np.round(sm_evals, 6).tolist())
print("wrote spectral_gold.edn")
