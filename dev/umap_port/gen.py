"""Gold-standard reference generator for the raster UMAP port.

Python+numba UMAP is the gold standard. Dumps to /tmp/umap_gold/:
  - the exact inputs to optimize_layout_euclidean,
  - the real (numba/float32) UMAP output,
  - a faithful float64 numpy reference (serial) replicating the Tausworthe RNG
    with int64-wraparound semantics (numba int64 == JVM long),
  - exact brute-force kNN (what UMAP uses for n<4096),
  - everything also as gold.edn / knn_gold.edn for the Clojure/Raster side.

Run from the umap source tree:  python3 dev/umap_port/gen.py
"""
import json, os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

import umap.umap_ as um
from umap.umap_ import fuzzy_simplicial_set, find_ab_params, make_epochs_per_sample
from umap.layouts import optimize_layout_euclidean

OUT = "/tmp/umap_gold"
SEED, N, DIM_IN, N_NEIGHBORS = 42, 300, 8, 15
N_EPOCHS, NEG_RATE, GAMMA, INIT_ALPHA, DIM = 200, 5.0, 1.0, 1.0, 2

rng = np.random.RandomState(SEED)
X, y = make_blobs(n_samples=N, n_features=DIM_IN, centers=4, random_state=SEED)
X = X.astype(np.float32)

fgraph, fsigmas, frhos = fuzzy_simplicial_set(
    X, N_NEIGHBORS, random_state=np.random.RandomState(SEED), metric="euclidean")
graph_raw = fgraph.tocoo()                       # symmetrized fuzzy graph (pre-prune)
graph = fgraph.tocoo(); graph.sum_duplicates()
graph.data[graph.data < (graph.data.max() / float(N_EPOCHS))] = 0.0
graph.eliminate_zeros()
head = graph.row.astype(np.int32); tail = graph.col.astype(np.int32)
weights = graph.data.astype(np.float64)
epochs_per_sample = make_epochs_per_sample(weights, N_EPOCHS)
a, b = find_ab_params(1.0, 0.1)
init = (rng.uniform(-10.0, 10.0, size=(N, DIM))).astype(np.float32)
rng_state = np.array([12345, 67890, 13579], dtype=np.int64)

# 1) REAL UMAP (numba, float32)
emb_umap = init.copy()
emb_umap = optimize_layout_euclidean(
    emb_umap, emb_umap, head, tail, N_EPOCHS, N, epochs_per_sample, a, b,
    rng_state.copy(), gamma=GAMMA, initial_alpha=INIT_ALPHA,
    negative_sample_rate=NEG_RATE, parallel=False, verbose=False, move_other=True)

# 2) faithful float64 reference replicating exact math + RNG
M32 = 0xFFFFFFFF
def _w64(x):
    x &= 0xFFFFFFFFFFFFFFFF
    return x - 0x10000000000000000 if x >= 0x8000000000000000 else x
def tau_rand_int(state):
    s0, s1, s2 = state
    s0 = _w64((((s0 & 4294967294) << 12) & M32) ^ ((((s0 << 13) & M32) ^ s0) >> 19))
    s1 = _w64((((s1 & 4294967288) << 4) & M32) ^ ((((s1 << 2) & M32) ^ s1) >> 25))
    s2 = _w64((((s2 & 4294967280) << 17) & M32) ^ ((((s2 << 3) & M32) ^ s2) >> 11))
    state[0], state[1], state[2] = s0, s1, s2
    r = (s0 ^ s1 ^ s2) & M32
    return r - 0x100000000 if r >= 0x80000000 else r
def clip(v): return 4.0 if v > 4.0 else (-4.0 if v < -4.0 else v)
def reference(emb0, do_neg, n_epochs):
    emb = emb0.astype(np.float64).copy()
    base = [int(v) for v in rng_state]
    bits = emb[:, 0].astype(np.float64).view(np.int64)
    states = [[_w64(base[c] + int(bits[i])) for c in range(3)] for i in range(N)]
    eps = epochs_per_sample.copy(); epn = eps / NEG_RATE
    eons = eps.copy(); eonsn = epn.copy(); alpha = INIT_ALPHA
    for ep in range(n_epochs):
        for i in range(eps.shape[0]):
            if eons[i] <= ep:
                j = int(head[i]); k = int(tail[i]); cur = emb[j]; oth = emb[k]
                d2 = float(np.sum((cur - oth) ** 2))
                gc = (-2.0*a*b*d2**(b-1.0))/(a*d2**b+1.0) if d2 > 0 else 0.0
                for dd in range(DIM):
                    g = clip(gc*(cur[dd]-oth[dd])); cur[dd] += g*alpha; oth[dd] += -g*alpha
                eons[i] += eps[i]
                if do_neg:
                    nn = int((ep - eonsn[i]) / epn[i])
                    for _ in range(nn):
                        k2 = tau_rand_int(states[j]) % N; o2 = emb[k2]
                        d2 = float(np.sum((cur - o2) ** 2))
                        if d2 > 0: gc = (2.0*GAMMA*b)/((0.001+d2)*(a*d2**b+1.0))
                        elif j == k2: continue
                        else: gc = 0.0
                        for dd in range(DIM):
                            g = clip(gc*(cur[dd]-o2[dd])) if gc > 0 else 0.0; cur[dd] += g*alpha
                    eonsn[i] += nn * epn[i]
        alpha = INIT_ALPHA * (1.0 - (float(ep) / float(n_epochs)))
    return emb
ref_attr_5 = reference(init, False, 5)
ref_attr   = reference(init, False, N_EPOCHS)
ref_full   = reference(init, True, N_EPOCHS)

# 3) exact brute-force kNN (float input matches what raster gets)
Xf = X.astype(np.float64)
nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="brute", metric="euclidean").fit(Xf)
knn_dst, knn_idx = nn.kneighbors(Xf)

os.makedirs(OUT, exist_ok=True)
for nm, arr in [("head", head), ("tail", tail), ("epochs_per_sample", epochs_per_sample),
                ("init", init.astype(np.float64)), ("emb_umap", emb_umap.astype(np.float64)),
                ("ref_attr_5epoch", ref_attr_5), ("ref_attr", ref_attr), ("ref_full", ref_full),
                ("labels", y), ("X", Xf), ("knn_idx", knn_idx), ("knn_dst", knn_dst)]:
    np.save(f"{OUT}/{nm}.npy", arr)

def evec(a): return "[" + " ".join(repr(float(x)) for x in np.asarray(a).ravel()) + "]"
def eivec(a): return "[" + " ".join(str(int(x)) for x in np.asarray(a).ravel()) + "]"
with open(f"{OUT}/gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim {DIM} :n-epochs {N_EPOCHS} :neg-rate {NEG_RATE} ")
    f.write(f":gamma {GAMMA} :init-alpha {INIT_ALPHA} :a {repr(float(a))} :b {repr(float(b))} ")
    f.write(f":n-edges {int(head.shape[0])} :rng-state {eivec(rng_state)} ")
    f.write(":head " + eivec(head) + " :tail " + eivec(tail) + " ")
    f.write(":eps " + evec(epochs_per_sample) + " :init " + evec(init.astype(np.float64)) + " ")
    f.write(":ref-attr-5 " + evec(ref_attr_5) + " :ref-attr " + evec(ref_attr) + "}")
with open(f"{OUT}/knn_gold.edn", "w") as f:
    f.write("{")
    f.write(f":n {N} :dim-in {DIM_IN} :k {N_NEIGHBORS} ")
    f.write(":X " + evec(Xf) + " :idx " + eivec(knn_idx) + " :dst " + evec(knn_dst) + "}")
with open(f"{OUT}/graph_gold.edn", "w") as f:
    # umap's smooth_knn_dist sigmas/rhos + the symmetrized fuzzy graph (COO, pre-prune)
    f.write("{")
    f.write(f":n {N} :k {N_NEIGHBORS} ")
    f.write(":sigmas " + evec(fsigmas) + " :rhos " + evec(frhos) + " ")
    f.write(":g-row " + eivec(graph_raw.row) + " :g-col " + eivec(graph_raw.col) + " ")
    f.write(":g-val " + evec(graph_raw.data) + "}")

meta = dict(N=N, DIM=DIM, DIM_IN=DIM_IN, N_NEIGHBORS=N_NEIGHBORS, N_EPOCHS=N_EPOCHS,
            NEG_RATE=NEG_RATE, GAMMA=GAMMA, INIT_ALPHA=INIT_ALPHA, a=float(a), b=float(b),
            n_edges=int(head.shape[0]), rng_state=[int(v) for v in rng_state])
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print(json.dumps(meta)); print("edges:", head.shape[0], "wrote ->", OUT)
