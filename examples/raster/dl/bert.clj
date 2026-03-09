(ns raster.dl.bert
  "BERT encoder inference using pre-trained weights from SafeTensors.

  Implements the standard BERT architecture:
    Embeddings (token + position + type) → N × TransformerBlock → pool

  Each TransformerBlock:
    x → MHA(Q,K,V,O) → add+LN → FFN(up,down) → add+LN

  Composes parametric ops from dl/nn and dl/attention for type-generic
  inference (float[] or double[]). Weight layout follows HuggingFace
  naming conventions."
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.dl.safetensors :as st]
            [clojure.data.json :as json]))

;; ================================================================
;; BERT Embeddings: token + position + token_type → LayerNorm
;; ================================================================

(deftm bert-embeddings (All [T]
  [token-emb :- (Array T) pos-emb :- (Array T) type-emb :- (Array T)
   ln-gamma :- (Array T) ln-beta :- (Array T)
   seq-len :- Long d-model :- Long eps :- Double]
  :- (Array T)
  (let [;; Sum: tok + pos + type (in-place on token-emb)
        _ (dotimes [i (* seq-len d-model)]
            (aset token-emb i (+ (aget token-emb i)
                                 (aget pos-emb i)
                                 (aget type-emb i))))]
    (nn/layer-norm token-emb ln-gamma ln-beta seq-len d-model eps))))

;; ================================================================
;; Residual add (deftm for compiler visibility)
;; ================================================================

(deftm residual-add! (All [T]
  [dst :- (Array T) src :- (Array T) n :- Long]
  :- (Array T)
  (dotimes [i n]
    (aset dst i (+ (aget dst i) (aget src i))))
  dst))

;; ================================================================
;; Encoder block (deftm — 20 params, at IFn limit)
;; ================================================================

(deftm bert-block (All [T]
  [x :- (Array T)
   wq :- (Array T) bq :- (Array T) wk :- (Array T) bk :- (Array T)
   wv :- (Array T) bv :- (Array T) wo :- (Array T) bo :- (Array T)
   attn-ln-w :- (Array T) attn-ln-b :- (Array T)
   fc-w :- (Array T) fc-b :- (Array T)
   proj-w :- (Array T) proj-b :- (Array T)
   ffn-ln-w :- (Array T) ffn-ln-b :- (Array T)
   seq-len :- Long hidden-size :- Long num-heads :- Long]
  :- (Array T)
  (let [n (* seq-len hidden-size)
        intermediate-size (* 4 hidden-size)
        ;; Self-attention
        attn-out (attn/multi-head-attention
                  x wq bq wk bk wv bv wo bo
                  seq-len hidden-size num-heads)
        ;; Residual + LayerNorm (post-attention)
        attn-out (residual-add! attn-out x n)
        x1 (nn/layer-norm attn-out attn-ln-w attn-ln-b
                          seq-len hidden-size 1e-12)
        ;; FFN: up → GELU → down
        ffn-up (nn/linear x1 fc-w fc-b seq-len hidden-size intermediate-size)
        ffn-up (nn/gelu ffn-up (* seq-len intermediate-size))
        ffn-down (nn/linear ffn-up proj-w proj-b
                            seq-len intermediate-size hidden-size)
        ;; Residual + LayerNorm (post-FFN)
        ffn-down (residual-add! ffn-down x1 n)]
    (nn/layer-norm ffn-down ffn-ln-w ffn-ln-b
                   seq-len hidden-size 1e-12))))

;; ================================================================
;; Mean pooling + L2 normalize (for sentence embeddings)
;; ================================================================

(deftm mean-pool (All [T]
  [x :- (Array T) seq-len :- Long dim :- Long]
  :- (Array T)
  (let [out (alloc-like x dim)
        inv-n (/ 1.0 (double seq-len))]
    (dotimes [s seq-len]
      (dotimes [d dim]
        (aset out d (+ (aget out d)
                       (aget x (+ (* s (int dim)) d))))))
    (dotimes [d dim]
      (aset out d (* (aget out d) inv-n)))
    out)))

(deftm l2-normalize (All [T]
  [v :- (Array T) n :- Long]
  :- (Array T)
  (let [norm (n/sqrt (loop [i 0 acc 0.0]
                       (if (< i n)
                         (recur (inc i) (+ acc (* (aget v i) (aget v i))))
                         acc)))
        inv-norm (/ 1.0 (n/max norm 1e-12))]
    (dotimes [i n]
      (aset v i (* (aget v i) inv-norm)))
    v)))

;; ================================================================
;; Public API
;; ================================================================

(defn load-model
  "Load a BERT model from a directory containing model.safetensors and config.json.
  Returns a model map with :weights, :config, and hyperparameters."
  [dir]
  (let [config (json/read-str (slurp (str dir "/config.json")) :key-fn keyword)
        weights (st/load-safetensors (str dir "/model.safetensors"))]
    {:weights weights
     :config config
     :hidden-size (:hidden_size config)
     :num-layers (:num_hidden_layers config)
     :num-heads (:num_attention_heads config)
     :intermediate-size (:intermediate_size config)
     :vocab-size (:vocab_size config)
     :max-position (:max_position_embeddings config)
     :layer-norm-eps (double (:layer_norm_eps config 1e-12))}))

(defn encode
  "Run BERT encoder forward pass. Uses defn (not deftm) because the layer
  loop pulls weights from a map — the block computation itself is the
  deftm `bert-block` above.
  model: from load-model
  token-ids: long[] of token IDs
  Returns hidden states [seq_len, hidden_size] as float[] (or double[])."
  [model ^longs token-ids]
  (let [{:keys [weights hidden-size num-layers num-heads
                vocab-size max-position layer-norm-eps]} model
        seq-len (clojure.core/alength token-ids)
        w (fn [^String name] (:data (get weights name)))
        dim (long hidden-size)
        tok-emb (nn/embedding (w "embeddings.word_embeddings.weight")
                              token-ids seq-len (long vocab-size) dim)
        pos-ids (long-array seq-len)
        _ (dotimes [i seq-len] (clojure.core/aset pos-ids i (long i)))
        pos-emb (nn/embedding (w "embeddings.position_embeddings.weight")
                              pos-ids seq-len (long max-position) dim)
        type-ids (long-array seq-len)
        type-emb (nn/embedding (w "embeddings.token_type_embeddings.weight")
                               type-ids seq-len 2 dim)
        x (bert-embeddings tok-emb pos-emb type-emb
                           (w "embeddings.LayerNorm.weight")
                           (w "embeddings.LayerNorm.bias")
                           seq-len hidden-size layer-norm-eps)]
    (loop [x x, layer 0]
      (if (clojure.core/< layer num-layers)
        (let [p (str "encoder.layer." layer ".")]
          (recur (bert-block x
                   (w (str p "attention.self.query.weight"))
                   (w (str p "attention.self.query.bias"))
                   (w (str p "attention.self.key.weight"))
                   (w (str p "attention.self.key.bias"))
                   (w (str p "attention.self.value.weight"))
                   (w (str p "attention.self.value.bias"))
                   (w (str p "attention.output.dense.weight"))
                   (w (str p "attention.output.dense.bias"))
                   (w (str p "attention.output.LayerNorm.weight"))
                   (w (str p "attention.output.LayerNorm.bias"))
                   (w (str p "intermediate.dense.weight"))
                   (w (str p "intermediate.dense.bias"))
                   (w (str p "output.dense.weight"))
                   (w (str p "output.dense.bias"))
                   (w (str p "output.LayerNorm.weight"))
                   (w (str p "output.LayerNorm.bias"))
                   seq-len hidden-size num-heads)
                 (clojure.core/inc layer)))
        x))))

(defn sentence-embedding
  "Compute a sentence embedding via mean pooling + L2 normalization.
  model: from load-model
  token-ids: long[] of token IDs
  Returns float[hidden_size] (or double[])."
  [model ^longs token-ids]
  (let [hidden (encode model token-ids)
        dim (:hidden-size model)
        seq-len (clojure.core/alength token-ids)]
    (l2-normalize (mean-pool hidden seq-len dim) dim)))

(deftm cosine-similarity (All [T]
  [a :- (Array T) b :- (Array T) n :- Long]
  :- Double
  (loop [i 0 acc 0.0]
    (if (< i n)
      (recur (+ i 1) (+ acc (* (aget a i) (aget b i))))
      acc))))
