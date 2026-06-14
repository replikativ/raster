"""Short-horizon umap dump: same inputs as gold, but n_epochs=3 WITH neg sampling.
Validates raster's f32 neg-sample path against real numba before chaos amplifies."""
import numpy as np, json, os
from umap.layouts import optimize_layout_euclidean

OUT = "/tmp/umap_gold"
meta = json.load(open(f"{OUT}/meta.json"))
N, DIM = meta["N"], meta["DIM"]
a, b = meta["a"], meta["b"]
GAMMA, INIT_ALPHA, NEG_RATE = meta["GAMMA"], meta["INIT_ALPHA"], meta["NEG_RATE"]
rng_state = np.array(meta["rng_state"], dtype=np.int64)

head = np.load(f"{OUT}/head.npy").astype(np.int32)
tail = np.load(f"{OUT}/tail.npy").astype(np.int32)
eps  = np.load(f"{OUT}/epochs_per_sample.npy").astype(np.float64)
init = np.load(f"{OUT}/init.npy").astype(np.float32)  # init was dumped as f64; cast back to f32

for EP in (1, 3, 10):
    emb = init.copy()
    emb = optimize_layout_euclidean(
        emb, emb, head, tail, EP, N, eps, a, b,
        rng_state.copy(), gamma=GAMMA, initial_alpha=INIT_ALPHA,
        negative_sample_rate=NEG_RATE, parallel=False, verbose=False, move_other=True)
    np.save(f"{OUT}/emb_umap_{EP}.npy", emb.astype(np.float64))
    print(f"wrote emb_umap_{EP}.npy  range=[{emb.min():.3f},{emb.max():.3f}]")
