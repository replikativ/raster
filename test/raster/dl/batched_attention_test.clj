(ns raster.dl.batched-attention-test
  "The BATCH axis of the training attention path: `rope` and `gqa-causal-mha` take a
  leading `batch` and operate on EXAMPLE-MAJOR rows (row = b·seq + s).

  The whole batching claim is one invariant, checked here from both ends:

    batched(stack of B examples) == stack of B single-example results        (forward)
    d/dθ of a batch loss == Σ_b d/dθ of the per-example losses               (backward)

  i.e. examples must NOT mix (causality is per example — the [seq,hd] slabs of
  different examples never see each other) and the gradient of a summed batch loss is
  exactly the accumulation of the per-example gradients. That second one is what makes
  a batched step a legitimate replacement for B accumulated single steps.

  Batching is a RELAYOUT, not a new kernel: gqa-causal-mha's batch axis is carried
  entirely by the shape literals handed to pack-heads / broadcast-kv-heads /
  batched-causal-sdpa / unpack-heads (head-major slab order, slab = h·batch + b — see
  the op's docstring). These tests are what pins that index algebra down."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attn]
            [raster.dl.loss]
            [raster.ad.reverse :as rev]))

(defn- rnd ^doubles [n ^long seed]
  (let [r (java.util.Random. seed) a (double-array n)]
    (dotimes [i n] (aset a i (.nextGaussian r)))
    a))

(defn- slice-rows
  "Rows [b*rows, b*rows+rows) of a row-major [_, cols] array."
  ^doubles [^doubles a ^long b ^long rows ^long cols]
  (let [out (double-array (* rows cols))]
    (System/arraycopy a (* b rows cols) out 0 (* rows cols))
    out))

(defn- max-abs-diff ^double [^doubles a ^doubles b]
  (reduce max 0.0 (map (fn [x y] (Math/abs (- (double x) (double y)))) a b)))

;; ---------------------------------------------------------------------------
;; Shapes: gemma-3-270m-ish but small — MQA (nkv=1 < nq=4) exercises the
;; broadcast-kv-heads / sum-kv-heads fan-out+fan-in under batching, which is the
;; part of the index algebra most likely to alias across examples.
;; ---------------------------------------------------------------------------
(def ^:private S 5)     ; seq
(def ^:private NQ 4)    ; query heads
(def ^:private NKV 2)   ; kv heads (GQA: group = 2)
(def ^:private HD 6)    ; head dim (even, for rope pairing)
(def ^:private B 3)     ; batch
(def ^:private THETA 10000.0)

(deftest rope-batched-equals-stacked-singles
  (testing "rope: position resets per example (row b*seq+s has position s, not b*seq+s)"
    (let [cols (* NQ HD)
          x (rnd (* B S cols) 42)
          got (attn/rope x B S NQ HD THETA)]
      (dotimes [b B]
        (let [xb (slice-rows x b S cols)
              want (attn/rope xb 1 S NQ HD THETA)
              gotb (slice-rows got b S cols)]
          (is (< (max-abs-diff gotb want) 1e-12)
              (str "rope example " b " must equal the single-example rope")))))))

(deftest rope-backward-batched-equals-stacked-singles
  (testing "rope-backward-dx batches the same way (the transpose rotation is per row)"
    (let [cols (* NQ HD)
          dy (rnd (* B S cols) 7)
          got (attn/rope-backward-dx dy B S NQ HD THETA)]
      (dotimes [b B]
        (let [dyb (slice-rows dy b S cols)
              want (attn/rope-backward-dx dyb 1 S NQ HD THETA)
              gotb (slice-rows got b S cols)]
          (is (< (max-abs-diff gotb want) 1e-12)
              (str "rope-backward example " b)))))))

(deftest gqa-causal-mha-batched-equals-stacked-singles
  (testing "batched GQA causal attention == B independent single-sequence attentions"
    ;; This is THE no-cross-example-leak gate: if the slab index algebra were wrong
    ;; (e.g. an example-major pack with a consecutive-repeat kv broadcast), example b
    ;; would attend to another example's keys and this would blow up.
    (let [qc (* NQ HD) kc (* NKV HD)
          Q (rnd (* B S qc) 1) K (rnd (* B S kc) 2) V (rnd (* B S kc) 3)
          got (attn/gqa-causal-mha Q K V B S NQ NKV HD)]
      (is (= (* B S qc) (alength got)) "output is [batch*seq, nq*hd]")
      (dotimes [b B]
        (let [want (attn/gqa-causal-mha (slice-rows Q b S qc)
                                        (slice-rows K b S kc)
                                        (slice-rows V b S kc)
                                        1 S NQ NKV HD)
              gotb (slice-rows got b S qc)]
          (is (< (max-abs-diff gotb want) 1e-12)
              (str "attention example " b " must equal the single-example attention")))))))

(deftest gqa-causal-mha-batched-does-not-leak-across-examples
  (testing "perturbing example 1's K/V leaves example 0's output bit-identical"
    (let [qc (* NQ HD) kc (* NKV HD)
          Q (rnd (* B S qc) 11) K (rnd (* B S kc) 12) V (rnd (* B S kc) 13)
          base (attn/gqa-causal-mha Q K V B S NQ NKV HD)
          K2 (aclone K) V2 (aclone V)]
      ;; wreck example 1 only
      (dotimes [i (* S kc)]
        (aset K2 (+ (* 1 S kc) i) 1000.0)
        (aset V2 (+ (* 1 S kc) i) -1000.0))
      (let [pert (attn/gqa-causal-mha Q K2 V2 B S NQ NKV HD)]
        (doseq [b [0 2]]
          (is (zero? (max-abs-diff (slice-rows base b S qc) (slice-rows pert b S qc)))
              (str "example " b " must be bit-identical when only example 1 changes")))
        (is (pos? (max-abs-diff (slice-rows base 1 S qc) (slice-rows pert 1 S qc)))
            "example 1 itself must have changed (the perturbation is real)")))))

;; ---------------------------------------------------------------------------
;; The gradient gate: a batch loss is a SUM over examples, so d/dQ of the batch is
;; the per-example gradients laid out example-major, and any SHARED-parameter grad
;; (here: none in raw attention — Q/K/V are per-example) accumulates. The shared-param
;; accumulation itself is covered end-to-end by the finetune SFT batch tests.
;; ---------------------------------------------------------------------------
(deftm gqa-batch-loss
  [Q :- (Array double) K :- (Array double) V :- (Array double)
   G :- (Array double)
   batch :- Long seq-len :- Long nq :- Long nkv :- Long hd :- Long] :- Double
  (raster.par/dot-product (raster.dl.attention/gqa-causal-mha Q K V batch seq-len nq nkv hd) G))

(deftest gqa-batched-vjp-equals-per-example-vjps
  (testing "VJP of the batched attention == the per-example VJPs, stacked"
    (let [qc (* NQ HD) kc (* NKV HD)
          Q (rnd (* B S qc) 21) K (rnd (* B S kc) 22) V (rnd (* B S kc) 23)
          G (rnd (* B S qc) 24)
          vg ((rev/value+grad #'gqa-batch-loss) Q K V G B S NQ NKV HD)
          dQ (nth vg 1) dK (nth vg 2) dV (nth vg 3)]
      (dotimes [b B]
        (let [vgb ((rev/value+grad #'gqa-batch-loss)
                   (slice-rows Q b S qc) (slice-rows K b S kc) (slice-rows V b S kc)
                   (slice-rows G b S qc) 1 S NQ NKV HD)]
          (is (< (max-abs-diff (slice-rows dQ b S qc) (nth vgb 1)) 1e-10)
              (str "dQ example " b))
          (is (< (max-abs-diff (slice-rows dK b S kc) (nth vgb 2)) 1e-10)
              (str "dK example " b " (the GQA kv fan-in must not cross examples)"))
          (is (< (max-abs-diff (slice-rows dV b S kc) (nth vgb 3)) 1e-10)
              (str "dV example " b)))))))
