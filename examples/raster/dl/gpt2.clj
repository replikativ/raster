(ns raster.dl.gpt2
  "GPT-2 inference using pre-trained weights from SafeTensors.

  Implements the standard GPT-2 architecture:
    Token+Position embeddings → N × TransformerBlock → LayerNorm → logits

  Each TransformerBlock (pre-norm):
    x → LN → CausalMHA(Q,K,V,O) → add → LN → FFN(up,down) → add

  Composes parametric ops from dl/nn and dl/attention for type-generic
  inference (float[] or double[]). Weight layout follows HuggingFace
  naming (Conv1D format, transposed at load time)."
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.dl.safetensors :as st]
            [clojure.data.json :as json]))

;; ================================================================
;; Weight transposition (GPT-2 Conv1D → standard linear)
;; ================================================================

(defn- transpose-weight
  "Transpose a float[] weight matrix from Conv1D [in, out] to linear [out, in]."
  ^floats [^floats w ^long rows ^long cols]
  (let [out (float-array (clojure.core/* rows cols))]
    (dotimes [i rows]
      (dotimes [j cols]
        (clojure.core/aset out
          (clojure.core/+ (clojure.core/* j rows) i)
          (clojure.core/aget w (clojure.core/+ (clojure.core/* i cols) j)))))
    out))

(defn- slice-array
  "Extract a contiguous slice from a float array."
  ^floats [^floats arr ^long offset ^long len]
  (let [out (float-array len)]
    (System/arraycopy arr (int offset) out 0 (int len))
    out))

;; ================================================================
;; Embedding lookup (deftm for typed dispatch)
;; ================================================================

(deftm gpt2-embeddings (All [T]
  [wte :- (Array T) wpe :- (Array T) token-ids :- (Array long)
   seq-len :- Long d-model :- Long]
  :- (Array T)
  (let [out (alloc-like wte (* seq-len d-model))]
    (dotimes [i seq-len]
      (let [tok-offset (* (aget token-ids i) d-model)
            pos-offset (* i d-model)
            out-offset (* i d-model)]
        (dotimes [d d-model]
          (aset out (+ out-offset d)
                (+ (aget wte (+ tok-offset d))
                   (aget wpe (+ pos-offset d)))))))
    out)))

;; ================================================================
;; Logits projection (weight-tied, deftm for typed dispatch)
;; ================================================================

(deftm gpt2-logits (All [T]
  [hidden :- (Array T) wte :- (Array T)
   seq-len :- Long d-model :- Long vocab-size :- Long]
  :- (Array T)
  (let [last-offset (* (- seq-len 1) d-model)
        out (alloc-like hidden vocab-size)]
    (dotimes [v vocab-size]
      (let [emb-offset (* v d-model)]
        (aset out v
              (loop [d 0 acc 0.0]
                (if (< d d-model)
                  (recur (+ d 1) (+ acc (* (aget hidden (+ last-offset d))
                                           (aget wte (+ emb-offset d)))))
                  acc)))))
    out)))

;; ================================================================
;; Residual add (deftm for typed dispatch + compiler visibility)
;; ================================================================

(deftm residual-add! (All [T]
  [dst :- (Array T) src :- (Array T) n :- Long]
  :- (Array T)
  (dotimes [i n]
    (aset dst i (+ (aget dst i) (aget src i))))
  dst))

;; ================================================================
;; Transformer block (deftm — 20 params, at IFn limit)
;; ================================================================

(deftm gpt2-block (All [T]
  [x :- (Array T)
   wq :- (Array T) bq :- (Array T) wk :- (Array T) bk :- (Array T)
   wv :- (Array T) bv :- (Array T) wo :- (Array T) bo :- (Array T)
   ln1-w :- (Array T) ln1-b :- (Array T)
   ln2-w :- (Array T) ln2-b :- (Array T)
   fc-w :- (Array T) fc-b :- (Array T)
   proj-w :- (Array T) proj-b :- (Array T)
   seq-len :- Long d-model :- Long n-head :- Long]
  :- (Array T)
  (let [n (* seq-len d-model)
        n-inner (* 4 d-model)
        ;; Pre-attention LayerNorm
        x-ln (nn/layer-norm x ln1-w ln1-b seq-len d-model 1e-5)
        ;; Causal multi-head attention
        attn-out (attn/causal-multi-head-attention
                  x-ln wq bq wk bk wv bv wo bo
                  seq-len d-model n-head)
        ;; Residual
        attn-out (residual-add! attn-out x n)
        ;; Pre-FFN LayerNorm
        x-ln2 (nn/layer-norm attn-out ln2-w ln2-b seq-len d-model 1e-5)
        ;; FFN: up → GELU → down
        ffn-up (nn/linear x-ln2 fc-w fc-b seq-len d-model n-inner)
        ffn-up (nn/gelu ffn-up (* seq-len n-inner))
        ffn-down (nn/linear ffn-up proj-w proj-b seq-len n-inner d-model)]
    ;; Residual
    (residual-add! ffn-down attn-out n))))

;; ================================================================
;; Public API
;; ================================================================

(defn load-model
  "Load a GPT-2 model from a directory containing model.safetensors and config.json.
  Returns a model map. Transposes Conv1D weights to standard linear layout."
  [dir]
  (let [config (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        raw-weights (st/load-safetensors (str dir "/model.safetensors"))
        n-embd (long (:n_embd config))
        n-inner (long (or (:n_inner config) (clojure.core/* 4 n-embd)))
        w (fn [^String name] (:data (get raw-weights name)))
        processed (into {}
                    (for [layer (range (:n_layer config))]
                      (let [p (str "h." layer ".")
                            c-attn-w (w (str p "attn.c_attn.weight"))
                            c-attn-b (w (str p "attn.c_attn.bias"))
                            c-attn-wt (transpose-weight c-attn-w n-embd (clojure.core/* 3 n-embd))
                            qkv-size (clojure.core/* n-embd n-embd)]
                        [layer
                         {:wq (slice-array c-attn-wt 0 qkv-size)
                          :wk (slice-array c-attn-wt qkv-size qkv-size)
                          :wv (slice-array c-attn-wt (clojure.core/* 2 qkv-size) qkv-size)
                          :bq (slice-array c-attn-b 0 n-embd)
                          :bk (slice-array c-attn-b n-embd n-embd)
                          :bv (slice-array c-attn-b (clojure.core/* 2 n-embd) n-embd)
                          :wo (transpose-weight (w (str p "attn.c_proj.weight")) n-embd n-embd)
                          :bo (w (str p "attn.c_proj.bias"))
                          :ln1-w (w (str p "ln_1.weight"))
                          :ln1-b (w (str p "ln_1.bias"))
                          :fc-w (transpose-weight (w (str p "mlp.c_fc.weight")) n-embd n-inner)
                          :fc-b (w (str p "mlp.c_fc.bias"))
                          :proj-w (transpose-weight (w (str p "mlp.c_proj.weight")) n-inner n-embd)
                          :proj-b (w (str p "mlp.c_proj.bias"))
                          :ln2-w (w (str p "ln_2.weight"))
                          :ln2-b (w (str p "ln_2.bias"))}])))]
    {:weights raw-weights
     :layers processed
     :wte (w "wte.weight")
     :wpe (w "wpe.weight")
     :ln-f-w (w "ln_f.weight")
     :ln-f-b (w "ln_f.bias")
     :config config
     :n-embd n-embd
     :n-layer (:n_layer config)
     :n-head (:n_head config)
     :n-inner n-inner
     :vocab-size (:vocab_size config)
     :max-position (:n_positions config)}))

(defn forward
  "Run GPT-2 forward pass. Returns hidden states [seq_len, n_embd] as float[].
  Uses defn (not deftm) because the layer loop pulls weights from a map —
  the block computation itself is the deftm `gpt2-block` above.
  model: from load-model
  token-ids: long[] of token IDs"
  [model ^longs token-ids]
  (let [{:keys [layers wte wpe ln-f-w ln-f-b
                n-embd n-layer n-head]} model
        seq-len (clojure.core/alength token-ids)
        x (gpt2-embeddings wte wpe token-ids seq-len n-embd)]
    (loop [x x, layer 0]
      (if (clojure.core/< layer n-layer)
        (let [{:keys [wq bq wk bk wv bv wo bo
                      ln1-w ln1-b ln2-w ln2-b
                      fc-w fc-b proj-w proj-b]} (get layers layer)]
          (recur (gpt2-block x wq bq wk bk wv bv wo bo
                             ln1-w ln1-b ln2-w ln2-b
                             fc-w fc-b proj-w proj-b
                             seq-len n-embd n-head)
                 (clojure.core/inc layer)))
        (nn/layer-norm x ln-f-w ln-f-b seq-len n-embd 1e-5)))))

(defn logits
  "Compute logits from hidden states by projecting through the token embedding
  matrix (weight tying). Returns float[vocab_size] for the last token."
  [model hidden ^long seq-len]
  (let [{:keys [wte n-embd vocab-size]} model]
    (gpt2-logits hidden wte seq-len n-embd vocab-size)))

(defn greedy-generate
  "Generate tokens greedily (argmax) from a prompt.
  model: from load-model
  token-ids: long[] prompt tokens
  n-tokens: number of tokens to generate
  Returns long[] of generated token IDs (prompt + generated)."
  [model ^longs token-ids ^long n-tokens]
  (loop [ids (vec (seq token-ids))
         remaining n-tokens]
    (if (clojure.core/<= remaining 0)
      (long-array ids)
      (let [input (long-array ids)
            hidden (forward model input)
            seq-len (count ids)
            lgt (logits model hidden seq-len)
            next-token (loop [i 1 best 0 best-score (aget lgt 0)]
                         (if (clojure.core/< i (alength lgt))
                           (let [s (aget lgt i)]
                             (if (> s best-score)
                               (recur (clojure.core/inc i) i s)
                               (recur (clojure.core/inc i) best best-score)))
                           best))]
        (recur (conj ids next-token) (clojure.core/dec remaining))))))
