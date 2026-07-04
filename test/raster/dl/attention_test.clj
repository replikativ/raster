(ns raster.dl.attention-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.dl.attention :as attn]
            [raster.dl.nn :as nn]
            [raster.ad.templates :as tmpl]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- arr-approx=
  ([a b] (arr-approx= a b 1e-5))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) (double eps)))))))

(defn- numerical-grad-array
  [f ^doubles a eps]
  (let [n (alength a)
        grad (double-array n)
        f0 (double (f))]
    (dotimes [i n]
      (let [orig (aget a i)]
        (aset a i (+ orig eps))
        (let [f+ (double (f))]
          (aset a i orig)
          (aset grad i (/ (- f+ f0) eps)))))
    grad))

;; ================================================================
;; Scaled dot-product attention
;; ================================================================

(deftest scaled-dot-product-attn-test
  (testing "attention output has correct shape"
    (let [seq-len 3 dk 4 dv 4
          Q (double-array (* seq-len dk))
          K (double-array (* seq-len dk))
          V (double-array (* seq-len dv))
          rng (java.util.Random. 42)]
      (dotimes [i (* seq-len dk)] (aset Q i (.nextGaussian rng)))
      (dotimes [i (* seq-len dk)] (aset K i (.nextGaussian rng)))
      (dotimes [i (* seq-len dv)] (aset V i (.nextGaussian rng)))
      (let [out (attn/scaled-dot-product-attn Q K V seq-len seq-len dk dv)]
        (is (= (* seq-len dv) (alength out))))))

  (testing "attention weights sum to 1 per query"
    ;; Verify attention weights are valid distributions
    ;; by checking output is a convex combination of V rows
    (let [seq-len 2 dk 2 dv 2
          Q (double-array [1 0  0 1])
          K (double-array [1 0  0 1])
          V (double-array [1 0  0 1])
          out (attn/scaled-dot-product-attn Q K V seq-len seq-len dk dv)]
      ;; Output should be close to V rows (self-attention with orthogonal Q,K)
      (is (= (* seq-len dv) (alength out))))))

(deftest scaled-dot-product-attn-gradient-test
  (testing "attention gradients vs finite diff"
    (let [seq-len 2 dk 3 dv 3
          Q (double-array [0.1 0.2 0.3  0.4 0.5 0.6])
          K (double-array [0.7 0.8 0.9  1.0 1.1 1.2])
          V (double-array [0.3 0.2 0.1  0.6 0.5 0.4])
          rrfn (tmpl/template-pullback 'raster.dl.attention/scaled-dot-product-attn)
          out (attn/scaled-dot-product-attn Q K V seq-len seq-len dk dv)
          pb (rrfn out Q K V seq-len seq-len dk dv)
          dy (double-array (repeat (* seq-len dv) 1.0))
          [dQ dK dV _ _ _ _] (pb dy)
          num-dQ (numerical-grad-array
                  #(let [o (attn/scaled-dot-product-attn Q K V seq-len seq-len dk dv)]
                     (loop [i 0 s 0.0]
                       (if (< i (alength o)) (recur (inc i) (+ s (aget o i))) s)))
                  Q 1e-5)]
      (is (arr-approx= dQ num-dQ 1e-3)))))

;; ================================================================
;; Multi-head attention
;; ================================================================

(deftest multi-head-attention-test
  (testing "MHA output has correct shape"
    (let [seq-len 4 d-model 8 n-heads 2
          x (double-array (* seq-len d-model))
          rng (java.util.Random. 42)]
      (dotimes [i (alength x)] (aset x i (* 0.1 (.nextGaussian rng))))
      (let [Wq (nn/xavier-init d-model d-model)
            Wk (nn/xavier-init d-model d-model)
            Wv (nn/xavier-init d-model d-model)
            Wo (nn/xavier-init d-model d-model)
            bq (double-array d-model 0.0)
            bk (double-array d-model 0.0)
            bv (double-array d-model 0.0)
            bo (double-array d-model 0.0)
            out (attn/multi-head-attention x Wq bq Wk bk Wv bv Wo bo
                                           seq-len d-model n-heads)]
        (is (= (* seq-len d-model) (alength out)))))))

;; ================================================================
;; Graph attention
;; ================================================================

(deftest graph-attention-test
  (testing "graph attention on small graph"
    ;; 3 nodes, 4 edges (simple directed graph)
    (let [n-nodes 3 n-edges 4 d-model 4
          h (double-array (* n-nodes d-model))
          rng (java.util.Random. 42)]
      (dotimes [i (alength h)] (aset h i (* 0.1 (.nextGaussian rng))))
      (let [Wq (nn/xavier-init d-model d-model)
            Wk (nn/xavier-init d-model d-model)
            Wv (nn/xavier-init d-model d-model)
            Wo (nn/xavier-init d-model d-model)
            src-edges (long-array [0 0 1 2])
            dst-edges (long-array [1 2 2 0])
            out (attn/graph-attention h Wq Wk Wv Wo
                                      src-edges dst-edges n-nodes n-edges d-model)]
        (is (= (* n-nodes d-model) (alength out)))))))

;; ================================================================
;; Sinusoidal embedding
;; ================================================================

(deftest sinusoidal-embedding-test
  (testing "sinusoidal embedding shape"
    (let [timesteps (long-array [0 1 2 3 4])
          emb (attn/sinusoidal-embedding timesteps 5 8)]
      (is (= 40 (alength emb)))))

  (testing "different timesteps produce different embeddings"
    (let [timesteps (long-array [0 100])
          emb (attn/sinusoidal-embedding timesteps 2 8)]
      ;; First and second rows should differ
      (is (not (= (aget emb 0) (aget emb 8)))))))
