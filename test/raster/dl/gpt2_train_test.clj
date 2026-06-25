(ns raster.dl.gpt2-train-test
  "End-to-end training test for a tiny GPT-2-shaped transformer.

  Exercises:
    - gpt2-embeddings (templated)
    - causal-multi-head-attention (refactored to slice-strided-2d / scatter-strided-2d)
    - layer-norm, linear, gelu, array-add (all templated)
    - cross-entropy-loss (templated)
    - Adam optimizer step over a Params tree

  Three layers of confidence:
    1. Lazy JIT forward — model evaluates and produces a finite scalar loss.
    2. Lazy value+grad — backward through the whole stack returns finite grads
       for every Param leaf.
    3. compile-aot fused train step — Adam updates run for ≥10 steps; loss
       decreases strictly between step 1 and step 10."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.params :as rp]
            [raster.dl.attention :as attn]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.array-ops :as ops]
            [raster.ad.reverse :as ad]))

;; Token + position embedding via two nn/embedding lookups summed:
;;   out[i,d] = wte[token_ids[i], d] + wpe[i, d]
;; Position indices are just [0,1,...,seq-len-1].
(defn- pos-ids ^longs [^long seq-len]
  (let [a (long-array seq-len)]
    (dotimes [i seq-len] (aset a i (long i)))
    a))

;; ----------------------------------------------------------------------
;; Tiny config — 1 layer, 8 d-model, 2 heads, 16 vocab, 8 seq-len.
;; ----------------------------------------------------------------------

(def cfg
  {:n-layer    1
   :d-model    8
   :n-head     2
   :vocab-size 16
   :seq-len    8
   :max-pos    16})

;; ----------------------------------------------------------------------
;; Loss model — single-layer GPT-2 with attention, MLP, layer-norms, tied
;; embedding. Weights live in a single Params tree.
;; ----------------------------------------------------------------------

(rp/defmodel tiny-gpt2-loss
  [w :- (Params (HMap :mandatory
                      {:wte    (Param (Array double))
                       :wpe    (Param (Array double))
                       :ln-f-w (Param (Array double))
                       :ln-f-b (Param (Array double))
                       :ub     (Frozen (Array double))     ;; unembed bias = zeros
                       :layer  (HMap :mandatory
                                     {:wq     (Param (Array double))
                                      :bq     (Param (Array double))
                                      :wk     (Param (Array double))
                                      :bk     (Param (Array double))
                                      :wv     (Param (Array double))
                                      :bv     (Param (Array double))
                                      :wo     (Param (Array double))
                                      :bo     (Param (Array double))
                                      :ln1-w  (Param (Array double))
                                      :ln1-b  (Param (Array double))
                                      :ln2-w  (Param (Array double))
                                      :ln2-b  (Param (Array double))
                                      :fc-w   (Param (Array double))
                                      :fc-b   (Param (Array double))
                                      :proj-w (Param (Array double))
                                      :proj-b (Param (Array double))})}))
   token-ids :- (Array long) position-ids :- (Array long)
   targets :- (Array long)
   seq-len :- Long d-model :- Long n-head :- Long vocab-size :- Long
   max-pos :- Long]
  :- Double
  (let [n          (clojure.core/* seq-len d-model)
        n-inner    (clojure.core/* 4 d-model)
        h-tok      (nn/embedding (:wte w) token-ids   seq-len vocab-size d-model)
        h-pos      (nn/embedding (:wpe w) position-ids seq-len max-pos    d-model)
        h-init     (ops/array-add h-tok h-pos n)
        ;; Pre-norm transformer block: ln1 → attn → residual → ln2 → mlp → residual
        h-ln1      (nn/layer-norm h-init (:ln1-w (:layer w)) (:ln1-b (:layer w))
                                  seq-len d-model 1e-5)
        h-attn     (attn/causal-multi-head-attention
                    h-ln1
                    (:wq (:layer w)) (:bq (:layer w))
                    (:wk (:layer w)) (:bk (:layer w))
                    (:wv (:layer w)) (:bv (:layer w))
                    (:wo (:layer w)) (:bo (:layer w))
                    seq-len d-model n-head)
        h-after-a  (ops/array-add h-init h-attn n)
        h-ln2      (nn/layer-norm h-after-a (:ln2-w (:layer w)) (:ln2-b (:layer w))
                                  seq-len d-model 1e-5)
        h-fc       (nn/linear h-ln2 (:fc-w (:layer w)) (:fc-b (:layer w))
                              seq-len d-model n-inner)
        h-act      (nn/gelu h-fc (clojure.core/* seq-len n-inner))
        h-proj     (nn/linear h-act (:proj-w (:layer w)) (:proj-b (:layer w))
                              seq-len n-inner d-model)
        h-after-m  (ops/array-add h-after-a h-proj n)
        ;; Final LN + tied unembed projection
        h-final    (nn/layer-norm h-after-m (:ln-f-w w) (:ln-f-b w)
                                  seq-len d-model 1e-5)
        logits     (nn/linear h-final (:wte w) (:ub w)
                              seq-len d-model vocab-size)]
    (loss/cross-entropy-loss logits targets seq-len vocab-size)))

;; ----------------------------------------------------------------------
;; Random init helpers
;; ----------------------------------------------------------------------

(defn- gaussian-arr ^doubles [^java.util.Random rng ^long n ^double scale]
  (let [a (double-array n)]
    (dotimes [i n] (aset a i (* scale (.nextGaussian rng))))
    a))

(defn- ones-arr ^doubles [^long n]
  (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))

(defn- init-weights [{:keys [d-model vocab-size max-pos]} ^long seed]
  (let [rng    (java.util.Random. seed)
        nin    (* 4 d-model)
        scale  (/ 1.0 (Math/sqrt (double d-model)))]
    {:wte    (gaussian-arr rng (* vocab-size d-model) scale)
     :wpe    (gaussian-arr rng (* max-pos d-model) scale)
     :ln-f-w (ones-arr d-model)
     :ln-f-b (double-array d-model)
     :ub     (double-array vocab-size)
     :layer  {:wq     (gaussian-arr rng (* d-model d-model) scale)
              :bq     (double-array d-model)
              :wk     (gaussian-arr rng (* d-model d-model) scale)
              :bk     (double-array d-model)
              :wv     (gaussian-arr rng (* d-model d-model) scale)
              :bv     (double-array d-model)
              :wo     (gaussian-arr rng (* d-model d-model) scale)
              :bo     (double-array d-model)
              :ln1-w  (ones-arr d-model)
              :ln1-b  (double-array d-model)
              :ln2-w  (ones-arr d-model)
              :ln2-b  (double-array d-model)
              :fc-w   (gaussian-arr rng (* d-model nin) scale)
              :fc-b   (double-array nin)
              :proj-w (gaussian-arr rng (* nin d-model) scale)
              :proj-b (double-array d-model)}}))

(defn- random-tokens [{:keys [vocab-size seq-len]} ^long seed]
  (let [rng (java.util.Random. seed)
        a (long-array seq-len)]
    (dotimes [i seq-len] (aset a i (long (.nextInt rng vocab-size))))
    a))

;; ----------------------------------------------------------------------
;; Tests
;; ----------------------------------------------------------------------

(deftest tiny-gpt2-lazy-jit-forward
  (testing "Lazy JIT forward — produces a finite scalar loss"
    (let [w  (init-weights cfg 42)
          tk (random-tokens cfg 1)
          tg (random-tokens cfg 2)
          {:keys [seq-len d-model n-head vocab-size max-pos]} cfg
          pi (pos-ids seq-len)
          loss (tiny-gpt2-loss w tk pi tg seq-len d-model n-head vocab-size max-pos)]
      (is (Double/isFinite loss))
      (is (pos? loss) "Cross-entropy loss is positive")
      (println "  lazy forward loss:" (format "%.4f" loss)))))

(deftest ^:compiled tiny-gpt2-compile-aot-forward
  (testing "compile-aot forward — produces a finite scalar loss matching lazy"
    (let [w  (init-weights cfg 42)
          tk (random-tokens cfg 1)
          tg (random-tokens cfg 2)
          {:keys [seq-len d-model n-head vocab-size max-pos]} cfg
          pi (pos-ids seq-len)
          lazy-loss (tiny-gpt2-loss w tk pi tg seq-len d-model n-head vocab-size max-pos)
          fast      (rp/compile-aot #'tiny-gpt2-loss)
          fast-loss (fast w tk pi tg seq-len d-model n-head vocab-size max-pos)]
      (is (Double/isFinite fast-loss))
      (is (< (Math/abs (- lazy-loss fast-loss)) 1e-9)
          (format "compile-aot loss %.6f matches lazy %.6f" fast-loss lazy-loss)))))

(deftest tiny-gpt2-lazy-value+grad
  (testing "Lazy value+grad — backward through whole stack returns finite grads"
    (let [w  (init-weights cfg 42)
          tk (random-tokens cfg 1)
          tg (random-tokens cfg 2)
          {:keys [seq-len d-model n-head vocab-size max-pos]} cfg
          pi (pos-ids seq-len)
          vg-fn (ad/value+grad #'tiny-gpt2-loss--flat)
          flat (raster.compiler.core.params-flatten/flatten-value
                (-> #'tiny-gpt2-loss meta :raster.params/treedefs (get 'w) :spec)
                w)
          out  (apply vg-fn (concat flat [tk pi tg seq-len d-model n-head vocab-size max-pos]))
          loss (nth out 0)
          grads (rest out)]
      (is (Double/isFinite loss))
      ;; The grad vector has one slot per ORIGINAL flat-arg (all leaves +
      ;; all non-tree args). Non-doubles slots get nil (non-differentiable).
      (is (>= (count grads) (count flat))
          "At least one gradient per leaf, plus nils for non-diff scalar args")
      (is (every? (fn [g]
                    (or (nil? g)        ;; Frozen leaves / Long params get nil
                        (and (instance? (Class/forName "[D") g)
                             (every? #(Double/isFinite %) g))))
                  grads)
          "All Param grads are finite double arrays"))))

(deftest ^:compiled tiny-gpt2-compile-aot-train-step-converges
  (testing "compile-aot fused train step — loss decreases over 10 Adam steps"
    (let [w     (init-weights cfg 42)
          tk    (random-tokens cfg 1)
          tg    (random-tokens cfg 2)
          {:keys [seq-len d-model n-head vocab-size max-pos]} cfg
          pi    (pos-ids seq-len)
          ts    (rp/compile-train-step #'tiny-gpt2-loss)
          state ((:init-state ts) w)
          train-fn (:train-fn ts)
          losses (vec (for [step (range 1 11)]
                        (train-fn w (:m state) (:v state)
                                  tk pi tg seq-len d-model n-head vocab-size max-pos
                                  1.0          ;; max-grad-norm
                                  0.01         ;; lr
                                  0.9 0.999 1e-8
                                  step)))]
      (println "  losses:" (mapv #(format "%.4f" %) losses))
      (is (every? Double/isFinite losses))
      (is (< (last losses) (first losses))
          (format "loss[10]=%.4f should be < loss[1]=%.4f"
                  (last losses) (first losses))))))
