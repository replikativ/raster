"""Gold-standard for the EVoC clustering MST stage.

EVoC clusters the node embedding: build_kdtree (Euclidean) -> parallel_boruvka
over mutual-reachability distances d_mr(a,b)=max(core_a,core_b,rdist(a,b)), where
core = rdist (squared euclidean) to the min_samples-th neighbor. The MST is unique
for distinct weights, so a dense Prim MST in raster must reproduce this exact tree.

Run from the evoc tree:  python3 <repo>/dev/umap_port/evoc_mst_gold.py
"""
import json, os
import numpy as np
import numba
from sklearn.datasets import make_blobs
from evoc.clustering import build_kdtree, parallel_boruvka
from evoc.numba_kdtree import parallel_tree_query
from evoc.cluster_trees import (mst_to_linkage_tree, condense_tree, extract_leaves,
                                get_cluster_label_vector)

OUT = "/tmp/umap_gold"; os.makedirs(OUT, exist_ok=True)
SEED, N, DIM, MIN_SAMPLES = 3, 200, 2, 5
X, y = make_blobs(n_samples=N, centers=4, n_features=DIM, random_state=SEED)
X = X.astype(np.float32)

tree = build_kdtree(X)
dists, nbrs = parallel_tree_query(tree, X, k=MIN_SAMPLES + 1, output_rdist=True)
core = dists.T[-1].astype(np.float64)                       # rdist to min_samples-th nbr
edges = parallel_boruvka(tree, numba.get_num_threads(), min_samples=MIN_SAMPLES, reproducible=True)
mst_from = edges[:, 0].astype(np.int64)
mst_to = edges[:, 1].astype(np.int64)
mst_w = edges[:, 2].astype(np.float64)

def evec(a): return "[" + " ".join(repr(float(x)) for x in np.asarray(a).ravel()) + "]"
def eivec(a): return "[" + " ".join(str(int(x)) for x in np.asarray(a).ravel()) + "]"

# full condensation chain (feed EVoC's own sorted MST so kernel validation is
# isolated from MST tie-degeneracy)
MIN_CLUSTER_SIZE = 10
sorted_mst = edges[np.argsort(edges[:, 2])]
linkage = mst_to_linkage_tree(sorted_mst)                  # n_edges x 4
condensed = condense_tree(linkage, MIN_CLUSTER_SIZE)
leaves = extract_leaves(condensed)
labels = get_cluster_label_vector(condensed, leaves, 0.0, N)

with open(f"{OUT}/evoc_mst_gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim {DIM} :min-samples {MIN_SAMPLES} :min-cluster-size {MIN_CLUSTER_SIZE} ")
    f.write(":X " + evec(X.astype(np.float64)) + " ")
    f.write(":core " + evec(core) + " ")
    f.write(":mst-from " + eivec(mst_from) + " :mst-to " + eivec(mst_to) + " :mst-w " + evec(mst_w) + " ")
    # sorted MST (input to the linkage stage)
    f.write(":smst-from " + eivec(sorted_mst[:, 0].astype(np.int64)) + " ")
    f.write(":smst-to " + eivec(sorted_mst[:, 1].astype(np.int64)) + " ")
    f.write(":smst-delta " + evec(sorted_mst[:, 2]) + " ")
    # linkage tree
    f.write(":lk-left " + eivec(linkage[:, 0].astype(np.int64)) + " ")
    f.write(":lk-right " + eivec(linkage[:, 1].astype(np.int64)) + " ")
    f.write(":lk-delta " + evec(linkage[:, 2]) + " ")
    f.write(":lk-size " + eivec(linkage[:, 3].astype(np.int64)) + " ")
    # condensed tree
    f.write(":c-parent " + eivec(condensed.parent) + " ")
    f.write(":c-child " + eivec(condensed.child) + " ")
    f.write(":c-lambda " + evec(condensed.lambda_val) + " ")
    f.write(":c-size " + eivec(condensed.child_size) + " ")
    f.write(":leaves " + eivec(leaves) + " ")
    f.write(":labels " + eivec(labels) + "}")

np.save(f"{OUT}/evoc_X.npy", X.astype(np.float64))
np.save(f"{OUT}/evoc_labels.npy", y)
print(f"n={N} dim={DIM} min_samples={MIN_SAMPLES} mst_edges={edges.shape[0]}")
print("core[:5]:", np.round(core[:5], 4).tolist())
print("total MST weight:", round(float(mst_w.sum()), 4))
print("wrote evoc_mst_gold.edn")
