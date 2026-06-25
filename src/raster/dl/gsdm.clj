(ns raster.dl.gsdm
  "Graphically Structured Diffusion Model (GSDM) implementation.

  A score-based diffusion model that operates on graphical models — collections
  of typed random variables connected by a known dependency graph. The model
  learns to denoise variables while respecting conditional independence structure
  via sparse graph attention.

  Architecture:
    FlatEmbedder → TimestepMLP → N× [ResnetBlock → GraphAttention] → Unembed

  Key features:
  - **Marginalisation**: variables can be MARGINALIZED (excluded from the graph),
    enabling a single trained model to answer queries about any variable subset.
  - **Conditioning**: OBSERVED variables are clamped (no noise added), providing
    inpainting-style conditioning through the graph structure.
  - **Graph attention**: sparse multi-head attention restricted to graphical model
    edges, so variables only attend to their Markov blanket.

  Variable states:
    LATENT (0)       — denoised by the model
    OBSERVED (1)     — clamped, provides conditioning signal
    MARGINALIZED (2) — excluded from representation and graph
    INTERVENED (3)   — for causal do-calculus reasoning

  Reference: 'Graphically Structured Diffusion Models' (Gruber et al.)"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :as core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone alloc-like]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.ad.templates :as tmpl]
            [raster.dl.nn :as nn]
            [raster.linalg.blas :as blas]
            [raster.dl.attention :as attn]
            [raster.dl.diffusion :as diff]
            [raster.dl.loss :as loss]
            [raster.dl.array-ops :as array-ops]
            [raster.params :as rp]
            [raster.tree :as tree]))

;; ================================================================
;; Variable state constants
;; ================================================================

(def ^:const LATENT 0)
(def ^:const OBSERVED 1)
(def ^:const MARGINALIZED 2)
(def ^:const INTERVENED 3)
(def ^:const N-STATES 4)

;; ================================================================
;; Marginalization: remove variables from graph and reindex
;; ================================================================

(defn marginalize-graph
  "Remove MARGINALIZED variables from the graph, filter edges, and reindex.

  Variables with state=MARGINALIZED are excluded from the flat representation.
  Edges touching marginalized nodes are removed. Remaining edges are reindexed
  to the compact representation.

  Returns {:values double[], :spaces long[], :states long[],
           :src-edges long[], :dst-edges long[],
           :n-vars long, :n-edges long,
           :original-indices long[] (old index for each new index)}"
  [^doubles values ^longs spaces ^longs states
   ^longs src-edges ^longs dst-edges n-vars n-edges]
  (let [n-vars (int n-vars)
        n-edges (int n-edges)
        ;; Build compact index: old -> new (-1 for marginalized)
        idx-map (long-array n-vars)
        _ (java.util.Arrays/fill idx-map (long -1))
        new-n (loop [i 0 next-idx 0]
                (if (< i n-vars)
                  (if (not= MARGINALIZED (aget states i))
                    (do (aset idx-map i (long next-idx))
                        (recur (inc i) (inc next-idx)))
                    (recur (inc i) next-idx))
                  next-idx))
        ;; Filter variable arrays
        new-values (double-array new-n)
        new-spaces (long-array new-n)
        new-states (long-array new-n)
        orig-indices (long-array new-n)
        _ (loop [i 0 j 0]
            (when (< i n-vars)
              (if (not= MARGINALIZED (aget states i))
                (do (aset new-values j (aget values i))
                    (aset new-spaces j (aget spaces i))
                    (aset new-states j (aget states i))
                    (aset orig-indices j (long i))
                    (recur (inc i) (inc j)))
                (recur (inc i) j))))
        ;; Count valid edges first
        new-n-edges (loop [e 0 cnt 0]
                      (if (< e n-edges)
                        (let [s (aget src-edges e)
                              d (aget dst-edges e)]
                          (if (and (>= (aget idx-map s) 0)
                                   (>= (aget idx-map d) 0))
                            (recur (inc e) (inc cnt))
                            (recur (inc e) cnt)))
                        cnt))
        new-src (long-array new-n-edges)
        new-dst (long-array new-n-edges)
        _ (loop [e 0 j 0]
            (when (< e n-edges)
              (let [s (aget src-edges e)
                    d (aget dst-edges e)
                    ns (aget idx-map s)
                    nd (aget idx-map d)]
                (if (and (>= ns 0) (>= nd 0))
                  (do (aset new-src j ns)
                      (aset new-dst j nd)
                      (recur (inc e) (inc j)))
                  (recur (inc e) j)))))]
    {:values new-values :spaces new-spaces :states new-states
     :src-edges new-src :dst-edges new-dst
     :n-vars (long new-n) :n-edges (long new-n-edges)
     :original-indices orig-indices}))

(defn unmarginalize-values
  "Scatter compacted values back to full-size array.
  Positions not in original-indices are filled with fill-val (default 0.0)."
  ([^doubles compact-values ^longs original-indices full-n]
   (unmarginalize-values compact-values original-indices full-n 0.0))
  ([^doubles compact-values ^longs original-indices full-n fill-val]
   (let [out (double-array full-n)]
     (when (not= 0.0 fill-val)
       (java.util.Arrays/fill out (double fill-val)))
     (dotimes [i (alength compact-values)]
       (aset out (aget original-indices i) (aget compact-values i)))
     out)))

;; ================================================================
;; Data normalization
;; ================================================================

(defn fit-normalizer
  "Compute per-variable mean and std from training samples.
  samples: sequence of double-arrays, each [n-vars]
  Returns {:mean double[], :std double[]}."
  [samples n-vars]
  (let [n-vars (int n-vars)
        n (double (count samples))
        mean (double-array n-vars)
        std (double-array n-vars)]
    ;; Accumulate mean
    (doseq [^doubles s samples]
      (dotimes [i n-vars]
        (aset mean i (+ (aget mean i) (aget s i)))))
    (dotimes [i n-vars]
      (aset mean i (/ (aget mean i) n)))
    ;; Accumulate variance
    (doseq [^doubles s samples]
      (dotimes [i n-vars]
        (let [d (- (aget s i) (aget mean i))]
          (aset std i (+ (aget std i) (* d d))))))
    (dotimes [i n-vars]
      (aset std i (Math/sqrt (Math/max 1e-8 (/ (aget std i) n)))))
    {:mean mean :std std}))

(defn fit-normalizer-2pass
  "Two-pass normalizer matching the reference Python implementation.
  Pass 1: compute mean/std from raw samples, normalize.
  Pass 2: compute mean2/std2 from normalized samples.
  Returns {:mean double[], :std double[], :mean2 double[], :std2 double[]}."
  [samples n-vars]
  (let [pass1 (fit-normalizer samples n-vars)
        mean1 ^doubles (:mean pass1)
        std1 ^doubles (:std pass1)
        n-vars (int n-vars)
        ;; Normalize all samples with pass-1 stats
        normalized (mapv (fn [^doubles s]
                           (let [out (double-array n-vars)]
                             (dotimes [i n-vars]
                               (aset out i (/ (- (aget s i) (aget mean1 i))
                                              (aget std1 i))))
                             out))
                         samples)
        ;; Pass 2: stats on normalized data
        pass2 (fit-normalizer normalized n-vars)]
    {:mean mean1 :std std1
     :mean2 (:mean pass2) :std2 (:std pass2)}))

(defn normalize-values
  "Normalize values: (x - mean) / std."
  ^doubles [^doubles values ^doubles mean ^doubles std n-vars]
  (let [n-vars (int n-vars)
        out (double-array n-vars)]
    (dotimes [i n-vars]
      (aset out i (/ (- (aget values i) (aget mean i))
                     (aget std i))))
    out))

(defn denormalize-values
  "Denormalize values: x * std + mean."
  ^doubles [^doubles values ^doubles mean ^doubles std n-vars]
  (let [n-vars (int n-vars)
        out (double-array n-vars)]
    (dotimes [i n-vars]
      (aset out i (+ (* (aget values i) (aget std i))
                     (aget mean i))))
    out))

(defn normalize-2pass
  "Apply 2-pass normalization. Falls back to single-pass if :mean2 absent."
  ^doubles [^doubles values normalizer n-vars]
  (let [v (normalize-values values
                            ^doubles (:mean normalizer)
                            ^doubles (:std normalizer)
                            n-vars)]
    (if (:mean2 normalizer)
      (normalize-values v
                        ^doubles (:mean2 normalizer)
                        ^doubles (:std2 normalizer)
                        n-vars)
      v)))

(defn denormalize-2pass
  "Reverse 2-pass normalization (denorm2 first, then denorm1)."
  ^doubles [^doubles values normalizer n-vars]
  (let [v (if (:mean2 normalizer)
            (denormalize-values values
                                ^doubles (:mean2 normalizer)
                                ^doubles (:std2 normalizer)
                                n-vars)
            values)]
    (denormalize-values v
                        ^doubles (:mean normalizer)
                        ^doubles (:std normalizer)
                        n-vars)))

;; ================================================================
;; Graph construction utilities
;; ================================================================

(defn fully-connected-edges
  "Build fully-connected graph edges including self-loops (every node connects
  to every node including itself). Self-loops are required for the attention
  mechanism to allow nodes to attend to their own representation, enabling
  timestep-conditional gating between self-value and neighbor aggregation.
  Returns {:src-edges long[], :dst-edges long[], :n-edges long}."
  [n-vars]
  (let [n-vars (int n-vars)
        n-edges (* n-vars n-vars)
        src (long-array n-edges)
        dst (long-array n-edges)
        _ (let [idx (volatile! 0)]
            (dotimes [i n-vars]
              (dotimes [j n-vars]
                (aset src @idx (long i))
                (aset dst @idx (long j))
                (vswap! idx inc))))]
    {:src-edges src :dst-edges dst :n-edges n-edges}))

;; ================================================================
;; Problem generators (amortized inference — fresh samples each call)
;; ================================================================

(defn make-linear-gaussian-problem
  "Create a linear Gaussian inference problem for GSDM training.

  Generative model: z ~ N(0, prior-std), x_i ~ N(z, obs-std) for i=1..n-obs.
  The latent z is variable 0, observations are variables 1..n-obs.
  All variables are in space 0 (same real-valued space).

  Returns a function (gen-fn rng) that produces:
    {:values double[n-vars], :spaces long[n-vars], :states long[n-vars],
     :src-edges long[], :dst-edges long[], :n-vars long, :n-edges long}

  For conditioning: set states[0]=LATENT, states[1..]=OBSERVED.
  For unconditional: all states=LATENT."
  [n-obs & {:keys [prior-std obs-std] :or {prior-std 1.0 obs-std 0.5}}]
  (let [n-vars (inc n-obs)
        {:keys [src-edges dst-edges n-edges]} (fully-connected-edges n-vars)
        spaces (long-array n-vars)  ;; all space 0
        ;; Default: latent z, observed x_i
        default-states (long-array n-vars)]
    (aset default-states 0 LATENT)
    (dotimes [i n-obs]
      (aset default-states (inc i) OBSERVED))
    (fn gen-sample [^java.util.Random rng]
      (let [z (* prior-std (.nextGaussian rng))
            values (double-array n-vars)]
        (aset values 0 z)
        (dotimes [i n-obs]
          (aset values (inc i) (+ z (* obs-std (.nextGaussian rng)))))
        {:values values
         :spaces (aclone spaces)
         :states (aclone default-states)
         :src-edges src-edges
         :dst-edges dst-edges
         :n-vars n-vars
         :n-edges n-edges}))))

(defn make-bmf-problem
  "Create a Bayesian Matrix Factorization problem for GSDM training.

  Generative model (simplified, all continuous):
    R_ij ~ N(0, 1) for i=1..t, j=1..n (selection weights)
    A_ik ~ N(0, 1) for i=1..t, k=1..m (factor matrix)
    E_jk = sum_i R_ij * A_ik  (observed matrix, n×m)

  Variable layout: [R (t*n), A (t*m), E (n*m)]
  All in space 0.  R and A are LATENT, E is OBSERVED.

  Returns a function (gen-fn rng) -> sample map."
  [t n m]
  (let [n-R (* t n)
        n-A (* t m)
        n-E (* n m)
        n-vars (+ n-R n-A n-E)
        {:keys [src-edges dst-edges n-edges]} (fully-connected-edges n-vars)
        spaces (long-array n-vars)
        default-states (long-array n-vars)]
    ;; R and A are LATENT
    (dotimes [i (+ n-R n-A)]
      (aset default-states i LATENT))
    ;; E is OBSERVED
    (dotimes [i n-E]
      (aset default-states (+ n-R n-A i) OBSERVED))
    (fn gen-sample [^java.util.Random rng]
      (let [values (double-array n-vars)
            ;; Generate R (t×n)
            R (double-array n-R)
            _ (dotimes [i n-R]
                (let [v (.nextGaussian rng)]
                  (aset R i v)
                  (aset values i v)))
            ;; Generate A (t×m)
            A (double-array n-A)
            _ (dotimes [i n-A]
                (let [v (.nextGaussian rng)]
                  (aset A i v)
                  (aset values (+ n-R i) v)))
            ;; Compute E = R^T * A  (n×m)
            ;; E_jk = sum_i R[i*n+j] * A[i*m+k]
            _ (dotimes [j n]
                (dotimes [k m]
                  (let [e-idx (+ n-R n-A (* j m) k)]
                    (loop [i 0, acc 0.0]
                      (if (< i t)
                        (recur (inc i)
                               (+ acc (* (aget R (+ (* i n) j))
                                         (aget A (+ (* i m) k)))))
                        (aset values e-idx acc))))))]
        {:values values
         :spaces (aclone spaces)
         :states (aclone default-states)
         :src-edges src-edges
         :dst-edges dst-edges
         :n-vars n-vars
         :n-edges n-edges}))))

;; ================================================================
;; Position embeddings
;; ================================================================

(defn compute-position-embeddings
  "Compute sinusoidal position embeddings for each variable within its space.
  Each variable's position is its index among variables sharing the same space.
  Returns double-array [n-vars * emb-dim]."
  ^doubles [^longs spaces n-vars emb-dim]
  (let [n-vars (int n-vars)
        emb-dim (int emb-dim)
        half-dim (quot emb-dim 2)
        out (double-array (* n-vars emb-dim))
        ;; Count position within each space
        space-counts (java.util.HashMap.)
        positions (long-array n-vars)
        log-scale (/ (Math/log 10000.0) (double (max 1 (dec half-dim))))]
    ;; Assign positions within each space
    (dotimes [i n-vars]
      (let [sp (aget spaces i)
            pos (.getOrDefault space-counts sp (long 0))]
        (aset positions i (long pos))
        (.put space-counts sp (long (inc pos)))))
    ;; Compute sinusoidal embeddings
    (dotimes [i n-vars]
      (let [pos (double (aget positions i))
            base (* i emb-dim)]
        (dotimes [d half-dim]
          (let [freq (Math/exp (* (- (double d)) log-scale))
                angle (* pos freq)]
            (aset out (+ base d) (Math/sin angle))
            (aset out (+ base half-dim d) (Math/cos angle))))))
    out))

;; ================================================================
;; Sinusoidal timestep embedding
;; ================================================================

(deftm embed-timestep
  "Sinusoidal timestep embedding for a single timestep.
  ref: reference array for type-directed allocation
  t: scalar timestep value (double)
  dim: embedding dimension
  Returns: array [dim]"
  (All [T] [ref :- (Array T) t :- Double dim :- Long] :- (Array T)
       (let [out (alloc-like ref dim)
             half-dim (quot dim 2)
             log-scale (/ (m/log 10000.0) (double (n/max 1 (dec half-dim))))]
         (dotimes [d half-dim]
           (let [freq (m/exp (* (- (double d)) log-scale))
                 angle (* t freq)]
             (aset out d (m/sin angle))
             (aset out (+ half-dim d) (m/cos angle))))
         out)))

;; ================================================================
;; sinusoidal-embed-backward: backward for embed-timestep
;; d_t = sum_d(dy[d]*freq[d]*cos(t*freq[d])) + sum_d(dy[half+d]*freq[d]*(-sin(t*freq[d])))
;; ================================================================

(deftm sinusoidal-embed-backward
  (All [T] [dy :- (Array T) t :- Double dim :- Long]
       :- Double
       (let [half-dim (int (quot dim 2))
             log-scale (/ (m/log 10000.0) (double (n/max 1 (dec half-dim))))]
         (loop [d 0 acc 0.0]
           (if (< d half-dim)
             (let [freq (m/exp (* (- (double d)) log-scale))
                   angle (* t freq)
              ;; d(sin(angle))/dt = cos(angle) * freq
              ;; d(cos(angle))/dt = -sin(angle) * freq
                   grad-sin (* (aget dy d) freq (m/cos angle))
                   grad-cos (* (aget dy (+ half-dim d)) freq (- (m/sin angle)))]
               (recur (inc d) (+ acc grad-sin grad-cos)))
             acc)))))

(tmpl/merge-into-template! 'raster.dl.gsdm/embed-timestep
                           {:pullback-factory (fn [_result ref t dim]
                                                (fn [dy]
                                                  (let [d-t (sinusoidal-embed-backward dy (double t) (long dim))]
                                                    [nil d-t nil])))})

;; ================================================================
;; TimestepMLP: sinusoidal → linear → SiLU → linear
;; ================================================================

(deftm timestep-mlp
  "Timestep MLP: embed → linear → SiLU → linear.
  t: scalar timestep
  W1: [emb_dim, emb_dim], b1: [emb_dim]
  W2: [emb_dim, emb_dim], b2: [emb_dim]
  Returns: [emb_dim]"
  (All [T] [t :- Double W1 :- (Array T) b1 :- (Array T)
            W2 :- (Array T) b2 :- (Array T) emb-dim :- Long]
       :- (Array T)
       (let [;; Sinusoidal embedding
             temb (embed-timestep W1 t emb-dim)
        ;; Linear 1
             h (nn/linear temb W1 b1 1 emb-dim emb-dim)
        ;; SiLU
             h (nn/silu h emb-dim)
        ;; Linear 2
             h (nn/linear h W2 b2 1 emb-dim emb-dim)]
         h)))

;; ================================================================
;; ResnetBlock: GroupNorm → SiLU → Linear → temb inject → GroupNorm → SiLU → Linear + skip
;; ================================================================

(deftm resnet-block
  "Residual block with timestep conditioning.
  x: [n_nodes, emb_dim] — node features
  temb: [emb_dim] — timestep embedding (already processed by TimestepMLP)
  W1: [emb_dim, emb_dim], b1: [emb_dim] — first linear
  W2: [emb_dim, emb_dim], b2: [emb_dim] — second linear
  Wt: [emb_dim, emb_dim], bt: [emb_dim] — timestep projection
  gamma1, beta1: [emb_dim] — first GroupNorm affine
  gamma2, beta2: [emb_dim] — second GroupNorm affine
  n-nodes, emb-dim: shape params
  Returns: [n_nodes, emb_dim]"
  (All [T] [x :- (Array T) temb :- (Array T)
            W1 :- (Array T) b1 :- (Array T)
            W2 :- (Array T) b2 :- (Array T)
            Wt :- (Array T) bt :- (Array T)
            gamma1 :- (Array T) beta1 :- (Array T)
            gamma2 :- (Array T) beta2 :- (Array T)
            n-nodes :- Long emb-dim :- Long]
       :- (Array T)
       (let [n (* n-nodes emb-dim)
        ;; GroupNorm needs channel-first layout [emb-dim, n-nodes] (PyTorch: b d 1 n)
        ;; Our data is node-major [n-nodes, emb-dim] — transpose around GN calls
             num-groups (n/max 1 (n/min 32 emb-dim))
        ;; GN1: transpose → group-norm → transpose back
             h-t (nn/transpose-2d x n-nodes emb-dim)
             h-t (nn/group-norm h-t gamma1 beta1 1 emb-dim n-nodes num-groups 1e-5)
             h (nn/transpose-2d h-t emb-dim n-nodes)
        ;; SiLU
             h (nn/silu h n)
        ;; Linear 1 (per-node: [n-nodes, emb-dim] × [emb-dim, emb-dim])
             h (nn/linear h W1 b1 n-nodes emb-dim emb-dim)
        ;; GN2 MUST precede the timestep injection: GroupNorm normalizes each
        ;; channel across nodes, and temb-proj is broadcast equally to every node
        ;; (constant along the normalized axis), so injecting BEFORE GN2 lets the
        ;; norm subtract it straight back out — making the model completely
        ;; time-blind. Inject AFTER the norm so the conditioning survives.
        ;; GN2: transpose → group-norm → transpose back
             h-t (nn/transpose-2d h n-nodes emb-dim)
             h-t (nn/group-norm h-t gamma2 beta2 1 emb-dim n-nodes num-groups 1e-5)
             h (nn/transpose-2d h-t emb-dim n-nodes)
        ;; Timestep injection AFTER the norm: h = h + broadcast(SiLU(linear(temb)))
             temb-proj (nn/silu temb emb-dim)
             temb-proj (nn/linear temb-proj Wt bt 1 emb-dim emb-dim)
             h (array-ops/broadcast-add h temb-proj n-nodes emb-dim)
        ;; SiLU
             h (nn/silu h n)
        ;; Linear 2
             h (nn/linear h W2 b2 n-nodes emb-dim emb-dim)
        ;; Skip connection: out = x + h
             out (array-ops/array-add x h n)]
         out)))

;; ================================================================
;; Multi-head graph attention (GSDM-style)
;; ================================================================

(deftm graph-attention-multihead
  "Multi-head graph attention using generic indexed-array primitives.
  h: [n_nodes, emb_dim] — node features
  Wq, bq, Wk, bk, Wv, bv: [emb_dim, emb_dim] + [emb_dim] — Q/K/V projections with bias
  gamma, beta: [emb_dim] — pre-attention GroupNorm
  src-edges, dst-edges: [n_edges] — edge lists
  n-nodes, n-edges, emb-dim, n-heads: shape params

  No output projection (Wo) — matches reference architecture.

  Composed from generic primitives:
    indexed-dot     → raw attention scores (Q·K per head)
    scale-clamp-exp → exp(clamp(score/√dk))
    scatter-add     → Z normalization sum
    scatter-mul-add → weighted value scatter
    segment-div     → normalize by Z"
  (All [T] [h :- (Array T)
            Wq :- (Array T) bq :- (Array T)
            Wk :- (Array T) bk :- (Array T)
            Wv :- (Array T) bv :- (Array T)
            gamma :- (Array T) beta :- (Array T)
            src-edges :- (Array long) dst-edges :- (Array long)
            n-nodes :- Long n-edges :- Long emb-dim :- Long n-heads :- Long]
       :- (Array T)
       (let [num-groups (n/max 1 (n/min 32 emb-dim))
             dk (quot emb-dim n-heads)
        ;; Pre-attention GroupNorm (channel-first layout for cross-node normalization)
             h-t (nn/transpose-2d h n-nodes emb-dim)
             h-t (nn/group-norm h-t gamma beta 1 emb-dim n-nodes num-groups 1e-5)
             h-norm (nn/transpose-2d h-t emb-dim n-nodes)
        ;; Project Q, K, V (with bias, matching reference)
             Q (nn/linear h-norm Wq bq n-nodes emb-dim emb-dim)
             K (nn/linear h-norm Wk bk n-nodes emb-dim emb-dim)
             V (nn/linear h-norm Wv bv n-nodes emb-dim emb-dim)
        ;; Raw attention scores: Q[dst] · K[src] per edge/head
             raw-scores (array-ops/indexed-dot Q K dst-edges src-edges
                                               n-nodes n-nodes n-edges dk emb-dim n-heads)
        ;; Scale + clamp + exp
             scores (array-ops/scale-clamp-exp raw-scores
                                               (/ 1.0 (n/sqrt (double dk))) 5.0
                                               (* n-edges n-heads))
        ;; Sum scores per destination node for Z normalization
             Z (array-ops/scatter-add scores dst-edges n-nodes n-edges n-heads)
        ;; Weighted scatter: score * V[src] → accumulated at dst
             wV (array-ops/scatter-mul-add scores V dst-edges src-edges
                                           n-nodes n-nodes n-edges dk emb-dim n-heads)
        ;; Normalize by Z
             wV (array-ops/segment-div wV Z n-nodes emb-dim n-heads 1e-6)
        ;; Residual connection (no output projection — matches reference)
             result (array-ops/array-add h wV (* n-nodes emb-dim))]
         result)))

;; ================================================================
;; TransformerBlock = ResnetBlock + GraphAttention
;; (Cannot be a single deftm due to Clojure's 20-param limit.
;;  Use transformer-block fn which calls both deftm sub-ops.)
;; ================================================================

(defn transformer-block
  "Single GSDM transformer block: ResnetBlock → GraphAttention.
  Uses deftm sub-operations for the heavy compute paths.

  x: [n_nodes, emb_dim], temb: [emb_dim]
  res-weights: {:W1 :b1 :W2 :b2 :Wt :bt :g1 :bn1 :g2 :bn2}
  attn-weights: {:Wq :bq :Wk :bk :Wv :bv :g :b}
  Returns: [n_nodes, emb_dim]"
  [x temb res-weights attn-weights
   src-edges dst-edges n-nodes n-edges emb-dim n-heads]
  (let [;; ResnetBlock (deftm — full AD support)
        h (resnet-block x temb
                        (:W1 res-weights) (:b1 res-weights)
                        (:W2 res-weights) (:b2 res-weights)
                        (:Wt res-weights) (:bt res-weights)
                        (:g1 res-weights) (:bn1 res-weights)
                        (:g2 res-weights) (:bn2 res-weights)
                        n-nodes emb-dim)
        ;; Graph attention (deftm — full AD support)
        h (graph-attention-multihead h
                                     (:Wq attn-weights) (:bq attn-weights)
                                     (:Wk attn-weights) (:bk attn-weights)
                                     (:Wv attn-weights) (:bv attn-weights)
                                     (:g attn-weights) (:b attn-weights)
                                     src-edges dst-edges
                                     n-nodes n-edges emb-dim n-heads)]
    h))

;; ================================================================
;; TransformerBlock as a tree/scan-vec step (defmodel)
;; ================================================================
;; The per-layer fold step for `tree/scan-vec`. Takes the running hidden
;; state `h` plus one layer's weight sub-tree (the SAME HMap shape as
;; `layer-spec`) and the shared extras. scan-vec splices each layer element's
;; leaves into this defmodel's --flat var (canonical key order, so the
;; ordering aligns automatically with the outer HVec element spec).
;;
;; NOTE: this layer spec must stay identical to `layer-spec` below — they are
;; declared in two places (this defmodel literal + weights-spec's HVec element).
(rp/defmodel transformer-block-step
  [h :- (Array double)
   layer :- (Params (HMap :mandatory
                          {:res-W1  (Param (Array double))  :res-b1  (Param (Array double))
                           :res-W2  (Param (Array double))  :res-b2  (Param (Array double))
                           :res-Wt  (Param (Array double))  :res-bt  (Param (Array double))
                           :res-g1  (Param (Array double))  :res-bn1 (Param (Array double))
                           :res-g2  (Param (Array double))  :res-bn2 (Param (Array double))
                           :attn-Wq (Param (Array double))  :attn-bq (Param (Array double))
                           :attn-Wk (Param (Array double))  :attn-bk (Param (Array double))
                           :attn-Wv (Param (Array double))  :attn-bv (Param (Array double))
                           :attn-g  (Param (Array double))  :attn-b  (Param (Array double))}))
   temb :- (Array double)
   src-edges :- (Array long) dst-edges :- (Array long)
   n-vars :- Long n-edges :- Long emb-dim :- Long n-heads :- Long]
  :- (Array double)
  (let [h-res (resnet-block h temb
                            (:res-W1 layer) (:res-b1 layer)
                            (:res-W2 layer) (:res-b2 layer)
                            (:res-Wt layer) (:res-bt layer)
                            (:res-g1 layer) (:res-bn1 layer)
                            (:res-g2 layer) (:res-bn2 layer)
                            n-vars emb-dim)]
    (graph-attention-multihead h-res
                               (:attn-Wq layer) (:attn-bq layer)
                               (:attn-Wk layer) (:attn-bk layer)
                               (:attn-Wv layer) (:attn-bv layer)
                               (:attn-g  layer) (:attn-b  layer)
                               src-edges dst-edges
                               n-vars n-edges emb-dim n-heads)))

;; ================================================================
;; FlatEmbedder: embed variable values + space + position
;; ================================================================

(deftm flat-embed
  "Embed flat variable values into emb_dim space.
  Combines value projection, space embedding, state embedding, and position embedding.
  values: [n_vars] — variable values (normalized)
  space-emb: [n_spaces * emb_dim] — learned space embeddings
  spaces: [n_vars] — space index per variable
  state-emb: [n_states * emb_dim] — learned state embeddings
  states: [n_vars] — variable state (LATENT/OBSERVED/INTERVENED)
  pos-emb: [n_vars * emb_dim] — precomputed sinusoidal position embeddings
  We: [emb_dim] — value projection weights
  be: [emb_dim] — value projection bias
  Returns: [n_vars * emb_dim]"
  (All [T] [values :- (Array T)
            space-emb :- (Array T) spaces :- (Array long)
            state-emb :- (Array T) states :- (Array long)
            pos-emb :- (Array T)
            We :- (Array T) be :- (Array T)
            n-vars :- Long emb-dim :- Long n-spaces :- Long n-states :- Long]
       :- (Array T)
       (array-ops/flat-embed-op values space-emb spaces state-emb states pos-emb
                                We be n-vars emb-dim n-spaces n-states)))

(deftm flat-unembed
  "Unembed from emb_dim back to scalar values.
  emb: [n_vars, emb_dim]
  Wu1: [emb_dim, emb_dim], bu1: [emb_dim] — unembed linear 1
  Wu2: [1, emb_dim], bu2: [1] — unembed linear 2 (project to scalar)
  n-vars, emb-dim: shape params
  Returns: [n_vars]"
  (All [T] [emb :- (Array T)
            Wu1 :- (Array T) bu1 :- (Array T)
            Wu2 :- (Array T) bu2 :- (Array T)
            n-vars :- Long emb-dim :- Long]
       :- (Array T)
       (let [;; Linear 1 + SiLU
             h (nn/linear emb Wu1 bu1 n-vars emb-dim emb-dim)
             h (nn/silu h (* n-vars emb-dim))
        ;; Linear 2: project to scalar per variable
             out (array-ops/dot-rows h Wu2 bu2 n-vars emb-dim)]
         out)))

;; ================================================================
;; Full GSDM model
;; ================================================================

(defn make-gsdm-config
  "Create GSDM model configuration.
  Returns a map describing the architecture."
  [& {:keys [emb-dim n-heads n-layers n-spaces n-states timesteps]
      :or {emb-dim 64 n-heads 2 n-layers 12 n-spaces 4 n-states N-STATES timesteps 1000}}]
  {:emb-dim emb-dim
   :n-heads n-heads
   :n-layers n-layers
   :n-spaces n-spaces
   :n-states n-states
   :timesteps timesteps})

(defn init-gsdm-weights
  "Initialize all GSDM weights.
  Returns a map of keyword→double-array for all learnable parameters.

  Weight naming: :layer-{i}-{component} for transformer blocks,
  :temb-{W1,b1,W2,b2} for timestep MLP,
  :embed-{We,be,space,state} for embedder,
  :unembed-{Wu1,bu1,Wu2,bu2} for unembedder."
  [config]
  (let [{:keys [emb-dim n-heads n-layers n-spaces n-states]} config
        n-states (or n-states N-STATES)
        he-scale (fn [fan-in] (n/sqrt (/ 2.0 (double fan-in))))
        rng (java.util.Random. 42)
        rand-array (fn [n scale]
                     (let [arr (double-array n)]
                       (dotimes [i n]
                         (aset arr i (* scale (.nextGaussian rng))))
                       arr))
        zero-array (fn [n] (double-array n))
        d emb-dim
        scale-d (he-scale d)]
    (into {}
          (concat
        ;; Timestep MLP
           [[:temb-W1 (rand-array (* d d) scale-d)]
            [:temb-b1 (zero-array d)]
            [:temb-W2 (rand-array (* d d) scale-d)]
            [:temb-b2 (zero-array d)]]
        ;; Embedder
           [[:embed-We (rand-array d (he-scale 1))]
            [:embed-be (zero-array d)]
            [:embed-space (rand-array (* n-spaces d) (/ 1.0 (n/sqrt (double d))))]
            [:embed-state (rand-array (* n-states d) (/ 1.0 (n/sqrt (double d))))]]
        ;; Unembedder
           [[:unembed-Wu1 (rand-array (* d d) scale-d)]
            [:unembed-bu1 (zero-array d)]
            [:unembed-Wu2 (rand-array d (he-scale d))]
            [:unembed-bu2 (zero-array 1)]]
        ;; Transformer blocks
           (mapcat
            (fn [i]
              [;; ResnetBlock weights
               [(keyword (str "layer-" i "-res-W1")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-res-b1")) (zero-array d)]
               [(keyword (str "layer-" i "-res-W2")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-res-b2")) (zero-array d)]
               [(keyword (str "layer-" i "-res-Wt")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-res-bt")) (zero-array d)]
               [(keyword (str "layer-" i "-res-g1")) (let [a (double-array d)] (java.util.Arrays/fill a 1.0) a)]
               [(keyword (str "layer-" i "-res-bn1")) (zero-array d)]
               [(keyword (str "layer-" i "-res-g2")) (let [a (double-array d)] (java.util.Arrays/fill a 1.0) a)]
               [(keyword (str "layer-" i "-res-bn2")) (zero-array d)]
             ;; GraphAttention weights (Q/K/V with bias, no output projection)
               [(keyword (str "layer-" i "-attn-Wq")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-attn-bq")) (zero-array d)]
               [(keyword (str "layer-" i "-attn-Wk")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-attn-bk")) (zero-array d)]
               [(keyword (str "layer-" i "-attn-Wv")) (rand-array (* d d) scale-d)]
               [(keyword (str "layer-" i "-attn-bv")) (zero-array d)]
               [(keyword (str "layer-" i "-attn-g")) (let [a (double-array d)] (java.util.Arrays/fill a 1.0) a)]
               [(keyword (str "layer-" i "-attn-b")) (zero-array d)]])
            (range n-layers))))))

(defn count-parameters
  "Count total number of learnable parameters."
  [weights]
  (reduce + (map (fn [[_ arr]] (alength arr)) weights)))

;; Note: A static deftm for the full model exceeds Clojure's 20-param limit.
;; Use gsdm-forward-dynamic below for the full model.

;; ================================================================
;; Dynamic-layer GSDM forward (loop-based, supports arbitrary n-layers)
;; ================================================================

(defn gsdm-forward-dynamic
  "GSDM forward pass with dynamic number of layers.
  Uses plain Clojure dispatch (not deftm) for the layer loop.

  Handles marginalization: MARGINALIZED variables are excluded from the graph.
  The returned prediction is over the non-marginalized variables only.

  weights: map from init-gsdm-weights
  config: from make-gsdm-config
  values: [n_vars] (after marginalization if applicable)
  spaces: [n_vars] (long[])
  states: [n_vars] (long[])
  t: timestep (double)
  src/dst-edges: [n_edges] (long[])
  n-vars, n-edges: sizes (after marginalization)"
  [weights config values spaces states t src-edges dst-edges n-vars n-edges]
  (let [{:keys [emb-dim n-heads n-layers n-spaces n-states]} config
        n-states (or n-states N-STATES)
        ;; Compute position embeddings for current variable set
        pos-emb (compute-position-embeddings spaces n-vars emb-dim)
        ;; Embed
        h (flat-embed values (:embed-space weights) spaces
                      (:embed-state weights) states pos-emb
                      (:embed-We weights) (:embed-be weights)
                      n-vars emb-dim n-spaces n-states)
        ;; Timestep MLP
        temb (timestep-mlp t (:temb-W1 weights) (:temb-b1 weights)
                           (:temb-W2 weights) (:temb-b2 weights) emb-dim)]
    ;; Transformer blocks
    (let [h (loop [h h, i 0]
              (if (< i n-layers)
                (let [k (fn [suffix] (keyword (str "layer-" i "-" suffix)))
                      res-w {:W1 (get weights (k "res-W1")) :b1 (get weights (k "res-b1"))
                             :W2 (get weights (k "res-W2")) :b2 (get weights (k "res-b2"))
                             :Wt (get weights (k "res-Wt")) :bt (get weights (k "res-bt"))
                             :g1 (get weights (k "res-g1")) :bn1 (get weights (k "res-bn1"))
                             :g2 (get weights (k "res-g2")) :bn2 (get weights (k "res-bn2"))}
                      attn-w {:Wq (get weights (k "attn-Wq")) :bq (get weights (k "attn-bq"))
                              :Wk (get weights (k "attn-Wk")) :bk (get weights (k "attn-bk"))
                              :Wv (get weights (k "attn-Wv")) :bv (get weights (k "attn-bv"))
                              :g (get weights (k "attn-g")) :b (get weights (k "attn-b"))}]
                  (recur
                   (transformer-block h temb res-w attn-w
                                      src-edges dst-edges n-vars n-edges emb-dim n-heads)
                   (inc i)))
                h))]
      ;; Unembed
      (flat-unembed h (:unembed-Wu1 weights) (:unembed-bu1 weights)
                    (:unembed-Wu2 weights) (:unembed-bu2 weights)
                    n-vars emb-dim))))

;; ================================================================
;; DDPM sampling (inference)
;; ================================================================

(defn gsdm-sample
  "Generate samples via DDPM reverse diffusion.

  Handles marginalization: MARGINALIZED variables are excluded from the graph
  during inference and returned as fill-val in the output.

  weights: trained model weights
  config: model config
  spaces: [n_vars] space indices (full, before marginalization)
  states: [n_vars] variable states (LATENT/OBSERVED/MARGINALIZED)
  observed-values: [n_vars] values for observed variables
  src-edges, dst-edges: [n_edges] (full graph)
  n-vars, n-edges: full sizes
  betas: [timesteps] noise schedule
  Optional :normalizer {:mean double[], :std double[]} for denormalization.
  Returns: [n_vars] generated sample (full size, marginalized positions = 0)"
  [weights config spaces states observed-values
   src-edges dst-edges n-vars n-edges betas
   & {:keys [normalizer rng] :or {rng (java.util.Random.)}}]
  (let [timesteps (alength betas) ;; use actual schedule length, not config
        alphas-cumprod (diff/compute-alphas-cumprod betas)
        ;; Marginalize graph
        mg (marginalize-graph observed-values spaces states src-edges dst-edges n-vars n-edges)
        mg-values ^doubles (:values mg)
        mg-spaces ^longs (:spaces mg)
        mg-states ^longs (:states mg)
        mg-src ^longs (:src-edges mg)
        mg-dst ^longs (:dst-edges mg)
        mg-n-vars (long (:n-vars mg))
        mg-n-edges (long (:n-edges mg))
        orig-indices ^longs (:original-indices mg)
        ;; Normalize observed values if normalizer provided (2-pass)
        mg-normalizer (when normalizer
                        (let [mg-mean (double-array mg-n-vars)
                              mg-std (double-array mg-n-vars)]
                          (dotimes [i mg-n-vars]
                            (let [oi (aget orig-indices i)]
                              (aset mg-mean i (aget ^doubles (:mean normalizer) oi))
                              (aset mg-std i (aget ^doubles (:std normalizer) oi))))
                          (cond-> {:mean mg-mean :std mg-std}
                            (:mean2 normalizer)
                            (assoc :mean2
                                   (let [a (double-array mg-n-vars)]
                                     (dotimes [i mg-n-vars]
                                       (aset a i (aget ^doubles (:mean2 normalizer)
                                                       (aget orig-indices i))))
                                     a)
                                   :std2
                                   (let [a (double-array mg-n-vars)]
                                     (dotimes [i mg-n-vars]
                                       (aset a i (aget ^doubles (:std2 normalizer)
                                                       (aget orig-indices i))))
                                     a)))))
        mg-values (if mg-normalizer
                    (normalize-2pass mg-values mg-normalizer mg-n-vars)
                    mg-values)
        ;; Start from pure noise for latent, observed values for observed
        x (double-array mg-n-vars)]
    (dotimes [i mg-n-vars]
      (aset x i
            (if (== (aget mg-states i) OBSERVED)
              (aget mg-values i)
              (.nextGaussian ^java.util.Random rng))))
    ;; Reverse diffusion loop
    (loop [step (dec (int timesteps))]
      (when (>= step 0)
        (let [;; Clamp observed before prediction
              _ (dotimes [i mg-n-vars]
                  (when (== (aget mg-states i) OBSERVED)
                    (aset x i (aget mg-values i))))
              ;; Predict x0 from current x_t
              pred-x0 (gsdm-forward-dynamic weights config x mg-spaces mg-states
                                            (double step) mg-src mg-dst mg-n-vars mg-n-edges)
              ;; Clamp prediction at observed positions
              _ (dotimes [i mg-n-vars]
                  (when (== (aget mg-states i) OBSERVED)
                    (aset ^doubles pred-x0 i (aget mg-values i))))
              ;; DDPM reverse step coefficients
              alpha-t (aget ^doubles alphas-cumprod step)
              beta-t (aget ^doubles betas step)
              alpha-prev (if (pos? step) (aget ^doubles alphas-cumprod (dec step)) 1.0)
              coeff-x0 (/ (* (Math/sqrt alpha-prev) beta-t) (- 1.0 alpha-t))
              coeff-xt (/ (* (Math/sqrt (- 1.0 beta-t)) (- 1.0 alpha-prev)) (- 1.0 alpha-t))
              ;; fixedlarge variance (DDPM Table 2): sigma^2 = beta_t
              sigma (if (pos? step) (Math/sqrt beta-t) 0.0)]
          (dotimes [i mg-n-vars]
            (when (not= (aget mg-states i) OBSERVED)
              (let [mean (+ (* coeff-x0 (aget ^doubles pred-x0 i))
                            (* coeff-xt (aget x i)))
                    noise-i (if (pos? step) (* sigma (.nextGaussian ^java.util.Random rng)) 0.0)]
                (aset x i (+ mean noise-i))))))
        (recur (dec step))))
    ;; Denormalize if needed (reverse 2-pass: denorm2 first, then denorm1)
    (let [result (if mg-normalizer
                   (denormalize-2pass x mg-normalizer mg-n-vars)
                   x)]
      ;; Scatter back to full-size array
      (unmarginalize-values result orig-indices n-vars))))

;; ================================================================
;; Tree-based defmodel API (params/compile-train-step compatible)
;;
;; Composes the block primitives (flat-embed, timestep-mlp, resnet-block,
;; graph-attention-multihead, flat-unembed) into a single rp/defmodel that
;; takes a structured weight tree. The user gets compile-aot, value+grad
;; with structured grads, and compile-train-step for fused training — all
;; without param-ordering bookkeeping or custom code generation.
;; ================================================================

(defn layer-spec
  "HMap spec for one transformer block's weights."
  []
  '(HMap :mandatory
         {:res-W1  (Param (Array double))  :res-b1  (Param (Array double))
          :res-W2  (Param (Array double))  :res-b2  (Param (Array double))
          :res-Wt  (Param (Array double))  :res-bt  (Param (Array double))
          :res-g1  (Param (Array double))  :res-bn1 (Param (Array double))
          :res-g2  (Param (Array double))  :res-bn2 (Param (Array double))
          :attn-Wq (Param (Array double))  :attn-bq (Param (Array double))
          :attn-Wk (Param (Array double))  :attn-bk (Param (Array double))
          :attn-Wv (Param (Array double))  :attn-bv (Param (Array double))
          :attn-g  (Param (Array double))  :attn-b  (Param (Array double))}))

(defn weights-spec
  "HMap spec for the full GSDM weight tree at given n-layers."
  [n-layers]
  (let [ls (layer-spec)]
    (list 'HMap :mandatory
          {:temb-W1     '(Param (Array double))   :temb-b1     '(Param (Array double))
           :temb-W2     '(Param (Array double))   :temb-b2     '(Param (Array double))
           :embed-We    '(Param (Array double))   :embed-be    '(Param (Array double))
           :embed-space '(Param (Array double))   :embed-state '(Param (Array double))
           :unembed-Wu1 '(Param (Array double))   :unembed-bu1 '(Param (Array double))
           :unembed-Wu2 '(Param (Array double))   :unembed-bu2 '(Param (Array double))
           :layers      (list 'HVec (vec (repeat n-layers ls)))})))

(defn flat->tree
  "Convert init-gsdm-weights output (flat keyword-keyed) into the tree shape
  expected by the gsdm-loss defmodel."
  [weights n-layers]
  (let [layer-keys [:res-W1 :res-b1 :res-W2 :res-b2 :res-Wt :res-bt
                    :res-g1 :res-bn1 :res-g2 :res-bn2
                    :attn-Wq :attn-bq :attn-Wk :attn-bk
                    :attn-Wv :attn-bv :attn-g :attn-b]]
    {:temb-W1     (:temb-W1 weights)     :temb-b1     (:temb-b1 weights)
     :temb-W2     (:temb-W2 weights)     :temb-b2     (:temb-b2 weights)
     :embed-We    (:embed-We weights)    :embed-be    (:embed-be weights)
     :embed-space (:embed-space weights) :embed-state (:embed-state weights)
     :unembed-Wu1 (:unembed-Wu1 weights) :unembed-bu1 (:unembed-bu1 weights)
     :unembed-Wu2 (:unembed-Wu2 weights) :unembed-bu2 (:unembed-bu2 weights)
     :layers (mapv (fn [i]
                     (into {} (map (fn [k]
                                     [k (get weights
                                             (keyword (str "layer-" i "-" (name k))))]))
                           layer-keys))
                   (range n-layers))}))

(defn gsdm-loss-body
  "Generate the GSDM loss body S-expression. The layer stack is folded by
  `tree/scan-vec` over `(:layers w)` — the compiler unrolls it from the HVec
  length in the weights spec, so this body is INDEPENDENT of n-layers (it no
  longer generates the per-layer chain by hand). Only emb-dim/n-heads/n-spaces
  are baked as constants.

  The body assumes outer scope binds: w, values, spaces, target, states,
  pos-emb, src-edges, dst-edges, t, n-vars, n-edges (the defmodel arg vector)."
  [emb-dim n-heads n-spaces]
  `(~'let [~'emb-dim-val ~(long emb-dim)
           ~'n-heads-val ~(long n-heads)
           ~'n-spaces-val ~(long n-spaces)
           ~'n-states-val ~(long N-STATES)
           ~'h-init (raster.dl.gsdm/flat-embed
                     ~'values (:embed-space ~'w) ~'spaces
                     (:embed-state ~'w) ~'states ~'pos-emb
                     (:embed-We ~'w) (:embed-be ~'w)
                     ~'n-vars ~'emb-dim-val ~'n-spaces-val ~'n-states-val)
           ~'temb (raster.dl.gsdm/timestep-mlp
                   ~'t (:temb-W1 ~'w) (:temb-b1 ~'w)
                   (:temb-W2 ~'w) (:temb-b2 ~'w) ~'emb-dim-val)
           ~'h-final (raster.tree/scan-vec raster.dl.gsdm/transformer-block-step
                                           ~'h-init (:layers ~'w)
                                           ~'temb ~'src-edges ~'dst-edges
                                           ~'n-vars ~'n-edges ~'emb-dim-val ~'n-heads-val)
           ~'pred (raster.dl.gsdm/flat-unembed
                   ~'h-final
                   (:unembed-Wu1 ~'w) (:unembed-bu1 ~'w)
                   (:unembed-Wu2 ~'w) (:unembed-bu2 ~'w)
                   ~'n-vars ~'emb-dim-val)
           ~'loss (raster.dl.array-ops/masked-mse-loss
                   ~'pred ~'target ~'states ~'n-vars)]
          ~'loss))

(defn make-gsdm-loss
  "Eval a deftm for the given config and return its var. The var has
  structured-arg surface (takes a weights tree via the HMap arg type);
  compile-aot, value+grad, and compile-train-step pick up the same
  surface via tree metadata."
  [{:keys [n-layers emb-dim n-heads n-spaces]
    :or {n-layers 2 emb-dim 8 n-heads 2 n-spaces 1}}]
  (let [w-spec (weights-spec n-layers)
        body (gsdm-loss-body emb-dim n-heads n-spaces)
        sym  (symbol (str "gsdm-loss-" n-layers "L-" emb-dim "d-" n-heads "h-" n-spaces "s"))
        form `(raster.core/deftm ~sym
                [~'w :- ~w-spec
                 ~'values :- (~'Array ~'double)
                 ~'spaces :- (~'Array ~'long)
                 ~'target :- (~'Array ~'double)
                 ~'states :- (~'Array ~'long)
                 ~'pos-emb :- (~'Array ~'double)
                 ~'src-edges :- (~'Array ~'long)
                 ~'dst-edges :- (~'Array ~'long)
                 ~'t :- ~'Double
                 ~'n-vars :- ~'Long
                 ~'n-edges :- ~'Long]
                :- ~'Double
                ~body)]
    (binding [*ns* (the-ns 'raster.dl.gsdm)]
      (eval form))
    (find-var (symbol "raster.dl.gsdm" (str sym)))))
