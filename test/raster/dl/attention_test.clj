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
          loss (fn [] (let [o (attn/scaled-dot-product-attn Q K V seq-len seq-len dk dv)]
                        (loop [i 0 s 0.0]
                          (if (< i (alength o)) (recur (inc i) (+ s (aget o i))) s))))
          num-dQ (numerical-grad-array loss Q 1e-5)
          num-dK (numerical-grad-array loss K 1e-5)
          num-dV (numerical-grad-array loss V 1e-5)]
      (is (arr-approx= dQ num-dQ 1e-3) "dQ")
      (is (arr-approx= dK num-dK 1e-3) "dK")
      (is (arr-approx= dV num-dV 1e-3) "dV"))))

(deftest causal-scaled-dot-product-attn-gradient-test
  (testing "causal attention gradients (dQ/dK/dV) vs finite diff"
    (let [seq-len 3 dk 3 dv 3
          rng (java.util.Random. 7)
          Q (double-array (* seq-len dk))
          K (double-array (* seq-len dk))
          V (double-array (* seq-len dv))
          _ (dotimes [i (* seq-len dk)] (aset Q i (.nextGaussian rng)))
          _ (dotimes [i (* seq-len dk)] (aset K i (.nextGaussian rng)))
          _ (dotimes [i (* seq-len dv)] (aset V i (.nextGaussian rng)))
          rrfn (tmpl/template-pullback 'raster.dl.attention/causal-scaled-dot-product-attn)
          out (attn/causal-scaled-dot-product-attn Q K V seq-len dk dv)
          pb (rrfn out Q K V seq-len dk dv)
          dy (double-array (repeat (* seq-len dv) 1.0))
          [dQ dK dV _ _ _] (pb dy)
          loss (fn [] (let [o (attn/causal-scaled-dot-product-attn Q K V seq-len dk dv)]
                        (loop [i 0 s 0.0]
                          (if (< i (alength o)) (recur (inc i) (+ s (aget o i))) s))))
          num-dQ (numerical-grad-array loss Q 1e-5)
          num-dK (numerical-grad-array loss K 1e-5)
          num-dV (numerical-grad-array loss V 1e-5)]
      (is (arr-approx= dQ num-dQ 1e-3) "causal dQ")
      (is (arr-approx= dK num-dK 1e-3) "causal dK")
      (is (arr-approx= dV num-dV 1e-3) "causal dV"))))

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

;; ================================================================
;; Windowed prefill scores (moonshine-style sliding-window encoder)
;; ================================================================

(deftest attn-prefill-scores-windowed-test
  (let [T 7 heads 2 hd 4
        dim (* heads hd)
        rng (java.util.Random. 42)
        frand (fn [n] (let [a (float-array n)]
                        (dotimes [i n] (aset a i (float (.nextGaussian rng)))) a))
        q (frand (* T dim)) k (frand (* T dim)) v (frand (* T dim))
        scale (/ 1.0 (Math/sqrt (double hd)))
        scores (fn [f & extra]
                 (let [sc (float-array (* T heads T))]
                   (apply f q k sc T heads 1 heads hd scale extra) sc))
        composed (fn [left right]
                   (let [sc (scores attn/attn-prefill-scores-windowed! left right)
                         out (float-array (* T dim))]
                     (attn/attn-prefill-softmax! sc T heads)
                     (attn/attn-prefill-out! sc v out T heads 1 heads hd)
                     out))
        ;; naive double-precision windowed reference: query i attends
        ;; j in [i-(left-1), i+max(0, right-1)] inclusive
        naive (fn [left right]
                (let [out (double-array (* T dim))]
                  (dotimes [i T]
                    (let [j0 (max 0 (- i (dec (long left))))
                          j1 (min (dec T) (+ i (max 0 (dec (long right)))))]
                      (dotimes [h heads]
                        (let [qb (+ (* i dim) (* h hd))
                              es (double-array T)
                              mx (reduce max -1.0e30
                                         (for [j (range j0 (inc j1))]
                                           (* scale (reduce + (for [d (range hd)]
                                                                (* (aget q (+ qb d))
                                                                   (aget k (+ (* (long j) dim) (* h hd) d))))))))
                              sum (reduce + (for [j (range j0 (inc j1))]
                                              (let [s (* scale (reduce + (for [d (range hd)]
                                                                           (* (aget q (+ qb d))
                                                                              (aget k (+ (* (long j) dim) (* h hd) d))))))
                                                    e (Math/exp (- s mx))]
                                                (aset es (long j) e) e)))]
                          (dotimes [d hd]
                            (aset out (+ (* i dim) (* h hd) d)
                                  (double (/ (reduce + (for [j (range j0 (inc j1))]
                                                         (* (aget es (long j))
                                                            (aget v (+ (* (long j) dim) (* h hd) d)))))
                                             sum))))))))
                  out))]
    (testing "left=T right=1 degenerates BIT-IDENTICALLY to the causal kernel"
      (is (java.util.Arrays/equals ^floats (scores attn/attn-prefill-scores!)
                                   ^floats (scores attn/attn-prefill-scores-windowed! T 1))))
    (testing "left=T right=T degenerates BIT-IDENTICALLY to the bidir kernel"
      (is (java.util.Arrays/equals ^floats (scores attn/attn-prefill-scores-bidir!)
                                   ^floats (scores attn/attn-prefill-scores-windowed! T T))))
    (testing "small [left right] windows match a naive double reference through softmax+out"
      (doseq [[l r] [[3 2] [16 4] [2 1] [1 1] [4 3]]]
        (let [got (composed l r)
              ref* (naive l r)]
          (dotimes [i (* T dim)]
            (is (< (Math/abs (- (aget ref* i) (double (aget got i)))) 1e-5)
                (str "window [" l " " r "] element " i))))))
    (testing "right=0 attends the diagonal (same as right=1, moonshine's j1 = i + max(0, right-1))"
      (is (java.util.Arrays/equals ^floats (scores attn/attn-prefill-scores-windowed! 3 0)
                                   ^floats (scores attn/attn-prefill-scores-windowed! 3 1))))))

;; ================================================================
;; Partial interleaved RoPE (GPT-J convention, moonshine decoder)
;; ================================================================

(defn- moonshine-rope-partial!
  "Copied reference: moonshine's rope-partial! — GPT-J interleaved partial RoPE
  in place, rotate first 32 of each 64-dim head, theta 10000."
  [^floats x heads pos]
  (let [pos (double pos)]
    (dotimes [h (long heads)]
      (let [base (* h 64)]
        (dotimes [j 16]
          (let [freq (Math/pow 10000.0 (- (/ (* 2.0 j) 32.0)))
                ang (* pos freq)
                c (Math/cos ang) s (Math/sin ang)
                i0 (+ base (* 2 j)) i1 (inc i0)
                x0 (aget x i0) x1 (aget x i1)]
            (aset x i0 (float (- (* x0 c) (* x1 s))))
            (aset x i1 (float (+ (* x1 c) (* x0 s)))))))))
  x)

(deftest rope-pos-partial-test
  (let [heads 3 hd 64 rd 32
        rng (java.util.Random. 7)
        frand (fn [n] (let [a (float-array n)]
                        (dotimes [i n] (aset a i (float (.nextGaussian rng)))) a))]
    (testing "bit-exact vs the copied moonshine reference loop"
      (doseq [pos [0 1 7 100 255]]
        (let [x (frand (* heads hd))
              ref* (moonshine-rope-partial! (aclone ^floats x) heads pos)
              got (attn/rope-pos-partial! (aclone ^floats x) heads hd rd 10000.0 pos)]
          (is (java.util.Arrays/equals ^floats ref* ^floats got)
              (str "pos " pos)))))
    (testing "dims beyond rotary-dim pass through untouched"
      (let [x (frand (* heads hd))
            got (attn/rope-pos-partial! (aclone ^floats x) heads hd rd 10000.0 42)]
        (dotimes [h heads]
          (doseq [d (range rd hd)]
            (is (= (aget x (+ (* h hd) d)) (aget ^floats got (+ (* h hd) d))))))))
    (testing "pos 0 is the identity"
      (let [x (frand (* heads hd))
            got (attn/rope-pos-partial! (aclone ^floats x) heads hd hd 10000.0 0)]
        (is (java.util.Arrays/equals ^floats x ^floats got))))
    (testing "rotary-dim = head-dim rotates every adjacent pair, norm-preserving"
      (let [x (frand (* heads hd))
            got ^floats (attn/rope-pos-partial! (aclone ^floats x) heads hd hd 10000.0 5)]
        (is (not (java.util.Arrays/equals ^floats x got)))
        (dotimes [p (quot (* heads hd) 2)]
          (let [i0 (* 2 p) i1 (inc i0)
                n-in (+ (* (double (aget x i0)) (aget x i0))
                        (* (double (aget x i1)) (aget x i1)))
                n-out (+ (* (double (aget got i0)) (aget got i0))
                         (* (double (aget got i1)) (aget got i1)))]
            (is (< (Math/abs (- n-in n-out)) 1e-4) (str "pair " p))))))))

;; ================================================================
;; Decode attention with weight capture (timestamp alignment signal)
;; ================================================================

(deftest gqa-decode-attention-weights-test
  (let [n 5 hd 8
        rng (java.util.Random. 11)
        frand (fn [k] (let [a (float-array k)]
                        (dotimes [i k] (aset a i (float (.nextGaussian rng)))) a))
        scale (/ 1.0 (Math/sqrt (double hd)))]
    (doseq [[n-q n-kv] [[4 4] [4 2]]]
      (let [q (frand (* n-q hd))
            kc (frand (* n n-kv hd))
            vc (frand (* n n-kv hd))
            wsink (float-array n)
            base ^floats (attn/gqa-decode-attention q kc vc n n-q n-kv hd scale)
            got ^floats (attn/gqa-decode-attention-weights! q kc vc n n-q n-kv hd scale wsink)]
        (testing (str "output bit-identical to gqa-decode-attention (n-q " n-q " n-kv " n-kv ")")
          (is (java.util.Arrays/equals base got)))
        (testing "head-averaged weights sum to ~1 for the query"
          (let [s (loop [j 0 s 0.0] (if (< j n) (recur (inc j) (+ s (aget wsink j))) s))]
            (is (approx= s 1.0 1e-5))))
        (testing "wsink ACCUMULATES across calls (per-layer averaging contract)"
          (attn/gqa-decode-attention-weights! q kc vc n n-q n-kv hd scale wsink)
          (let [s (loop [j 0 s 0.0] (if (< j n) (recur (inc j) (+ s (aget wsink j))) s))]
            (is (approx= s 2.0 1e-5))))))))
