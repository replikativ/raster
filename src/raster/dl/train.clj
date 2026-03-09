(ns raster.dl.train
  "Training infrastructure for the Raster deep learning framework.

  Provides:
    data-loader       - batching utility
    collate-doubles   - stack samples into flat arrays
    train-epoch!      - train one epoch over a dataset"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.arrays :refer [aget aset alength aclone acopy!]]))

;; ================================================================
;; DataLoader
;; ================================================================

(defn data-loader
  "Returns a lazy seq of batches from dataset.
  dataset: vector of [x y] pairs (double arrays)
  batch-size: number of samples per batch
  Options:
    :shuffle? - randomize order (default true)
    :drop-last? - drop incomplete last batch (default false)"
  [dataset batch-size & {:keys [shuffle? drop-last?]
                         :or {shuffle? true drop-last? false}}]
  (let [n (count dataset)
        indices (if shuffle?
                  (shuffle (vec (range n)))
                  (vec (range n)))
        batches (if drop-last?
                  (partition batch-size indices)
                  (partition-all batch-size indices))]
    (map (fn [batch-indices]
           (let [items (mapv (fn [i] (nth dataset i)) batch-indices)]
             {:xs (mapv first items)
              :ys (mapv second items)
              :batch-size (count items)}))
         batches)))

;; ================================================================
;; Collate: stack batch items into single flat arrays
;; ================================================================

(defn collate-doubles
  "Stack a vector of double[] into a single flat double[batch*feature_dim].
  items: vector of double arrays, all same length"
  ^doubles [items]
  (let [batch (count items)
        dim (clojure.core/alength ^doubles (first items))
        out (double-array (* batch dim))]
    (dotimes [b batch]
      (acopy! ^doubles (nth items b) 0
              out (* b dim) dim))
    out))

(defn collate-longs
  "Stack a vector of long[] into a single flat long[batch*feature_dim].
  items: vector of long arrays, all same length"
  ^longs [items]
  (let [batch (count items)
        dim (clojure.core/alength ^longs (first items))
        out (long-array (* batch dim))]
    (dotimes [b batch]
      (acopy! ^longs (nth items b) 0
              out (* b dim) dim))
    out))

;; ================================================================
;; Training loop helper
;; ================================================================

(defn train-epoch!
  "Train one epoch. Returns average loss.
  model-fn: (fn [batch-data] -> loss) - forward + backward + update
  dataset: vector of samples
  batch-size: samples per batch
  Options:
    :shuffle? - randomize order (default true)
    :log-interval - print every N batches (default nil = no printing)"
  [model-fn dataset batch-size & {:keys [shuffle? log-interval]
                                  :or {shuffle? true}}]
  (let [batches (data-loader dataset batch-size :shuffle? shuffle?)
        n-batches (count (seq batches))]
    (loop [bs (seq batches) total-loss 0.0 n-seen 0 batch-idx 0]
      (if bs
        (let [batch (first bs)
              loss (double (model-fn batch))]
          (when (and log-interval (zero? (rem batch-idx log-interval)))
            (println (str "  batch " batch-idx "/" n-batches
                          " loss=" (format "%.4f" loss))))
          (recur (next bs) (+ total-loss loss) (inc n-seen) (inc batch-idx)))
        (if (pos? n-seen) (/ total-loss n-seen) 0.0)))))
