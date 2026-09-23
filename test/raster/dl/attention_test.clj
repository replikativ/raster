(ns raster.dl.attention-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.dl.attention :as attn]
            [raster.dl.attention-reference :as attention-reference]
            [raster.dl.array-ops :as ops]
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

(deftest bidirectional-sdpa-composes-layouts-contractions-and-softmax
  (let [rows 4 heads 2 head-dim 3 width (* heads head-dim)
        q (double-array (map #(/ (- (mod % 17) 8) 9.0) (range (* rows width))))
        k (double-array (map #(/ (- (mod % 13) 6) 8.0) (range (* rows width))))
        v (double-array (map #(/ (- (mod % 11) 5) 7.0) (range (* rows width))))
        scale (/ 1.0 (Math/sqrt (double head-dim)))
        scores (double-array (* rows heads rows))
        expected (double-array (* rows width))]
    (attn/attn-prefill-scores-bidir! q k scores rows heads 1 heads head-dim scale)
    (attn/attn-prefill-softmax! scores rows heads)
    (attn/attn-prefill-out! scores v expected rows heads 1 heads head-dim)
    (let [actual ^doubles (attn/bidirectional-sdpa q k v rows heads head-dim scale)]
      (is (arr-approx= expected actual 1.0e-12)
          "head-major batched contractions preserve the token-major staged API semantics"))))

(deftest packed-prefill-views-match-materialized-inputs
  (let [rows 3
        heads 2
        head-dim 4
        width (* heads head-dim)
        stride (+ (* 3 width) 2)
        q-offset 1
        v-offset (+ 1 (* 2 width))
        packed (double-array (map #(/ (- (mod % 19) 9) 7.0) (range (* rows stride))))
        q (double-array (* rows width))
        v (double-array (* rows width))
        q-reference (double-array (* rows width))
        q-strided (double-array (* rows width))
        scores (double-array (map #(/ (inc (mod % 5)) 15.0)
                                  (range (* rows heads rows))))
        out-reference (double-array (* rows width))
        out-strided (double-array (* rows width))]
    (dotimes [i (* rows width)]
      (let [row (quot i width)
            column (rem i width)]
        (aset q i (aget packed (+ (* row stride) q-offset column)))
        (aset v i (aget packed (+ (* row stride) v-offset column)))))
    (attn/rope-prefill! q q-reference rows heads head-dim 10000.0)
    (attn/rope-prefill-strided! packed q-strided rows heads head-dim 10000.0
                                stride q-offset)
    (let [half (quot head-dim 2)
          cosines (double-array
                   (for [row (range rows) i (range half)]
                     (Math/cos (* row (Math/pow 10000.0 (/ (* -2.0 i) head-dim))))))
          sines (double-array
                 (for [row (range rows) i (range half)]
                   (Math/sin (* row (Math/pow 10000.0 (/ (* -2.0 i) head-dim))))))
          table-out (double-array (* rows width))]
      (attn/rope-prefill-strided-table! packed cosines sines table-out
                                        rows heads head-dim stride q-offset)
      (is (arr-approx= q-reference table-out 1.0e-12)))
    (attn/attn-prefill-out! scores v out-reference rows heads 1 heads head-dim)
    (attn/attn-prefill-out-strided! scores packed out-strided
                                    rows heads 1 heads head-dim stride v-offset)
    (is (arr-approx= q-reference q-strided 1.0e-12))
    (is (arr-approx= out-reference out-strided 1.0e-12))))

(deftest head-major-prefill-masks
  (testing "the uniform mask follows strided-batch BLAS head-major layout"
    (let [nrows 4 heads 2
          scores (float-array (map float (range (* heads nrows nrows))))]
      (attn/attn-prefill-mask-windowed-head-major! scores nrows heads 2 2)
      (dotimes [h heads]
        (dotimes [i nrows]
          (dotimes [j nrows]
            (let [actual (aget scores (+ (* h nrows nrows) (* i nrows) j))]
              (if (< (Math/abs (long (- i j))) 2)
                (is (= (float (+ (* h nrows nrows) (* i nrows) j)) actual))
                (is (= (float -1.0e30) actual)))))))))
  (testing "segmented masking combines padding and a local window"
    (let [batch 2 nrows 4 heads 2 lengths (long-array [2 4])
          scores (float-array (repeat (* heads batch nrows nrows) 1.0))]
      (attn/attn-prefill-mask-segmented-head-major!
       scores lengths batch nrows heads 2 2)
      (dotimes [h heads]
        (dotimes [b batch]
          (dotimes [i nrows]
            (dotimes [j nrows]
              (let [idx (+ (* (+ (* h batch) b) nrows nrows) (* i nrows) j)
                    active (aget lengths b)
                    keep? (and (< i active) (< j active)
                               (< (Math/abs (long (- i j))) 2))]
                (is (= (float (if keep? 1.0 -1.0e30))
                       (aget scores idx)))))))))))

(deftest functional-prefill-softmax-matches-in-place-reference
  (doseq [[rows heads] [[1 1] [3 2] [7 3]]]
    (let [source (float-array
                  (map #(float (/ (- (mod (* 17 %) 29) 14) 3.0))
                       (range (* rows heads rows))))
          reference (aclone source)
          actual (attn/attn-prefill-softmax source rows heads)]
      (attn/attn-prefill-softmax! reference rows heads)
      (is (= (alength reference) (alength actual)))
      (is (every? (fn [i]
                    (< (Math/abs (- (double (aget reference i))
                                    (double (aget actual i))))
                       1.0e-6))
                  (range (alength reference)))))))

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

(defn- headwise-causal-mha-reference
  [x Wq bq Wk bk Wv bv Wo bo seq-len d-model n-heads]
  (let [dk (quot d-model n-heads)
        n (* seq-len d-model)
        Q (nn/linear x Wq bq seq-len d-model d-model)
        K (nn/linear x Wk bk seq-len d-model d-model)
        V (nn/linear x Wv bv seq-len d-model d-model)]
    (loop [head 0 acc (double-array n)]
      (if (< head n-heads)
        (let [column-offset (* head dk)
              Qh (ops/slice-strided-2d Q seq-len d-model column-offset dk)
              Kh (ops/slice-strided-2d K seq-len d-model column-offset dk)
              Vh (ops/slice-strided-2d V seq-len d-model column-offset dk)
              head-out (attn/causal-scaled-dot-product-attn Qh Kh Vh seq-len dk dk)
              wide (ops/scatter-strided-2d head-out seq-len d-model column-offset dk)]
          (recur (inc head) (ops/array-add acc wide n)))
        (nn/linear acc Wo bo seq-len d-model d-model)))))

(deftest batched-causal-mha-preserves-headwise-and-ragged-width-semantics
  (doseq [d-model [4 5]]
    (let [seq-len 2 n-heads 2
          values (fn [n offset]
                   (double-array (map #(/ (+ offset %) 37.0) (range n))))
          x (values (* seq-len d-model) -3)
          Wq (values (* d-model d-model) -7)
          Wk (values (* d-model d-model) 2)
          Wv (values (* d-model d-model) -5)
          Wo (values (* d-model d-model) 1)
          bq (values d-model -2) bk (values d-model 1)
          bv (values d-model -1) bo (values d-model 3)
          expected (headwise-causal-mha-reference
                    x Wq bq Wk bk Wv bv Wo bo seq-len d-model n-heads)
          actual (attn/causal-multi-head-attention
                  x Wq bq Wk bk Wv bv Wo bo seq-len d-model n-heads)]
      (is (= (* seq-len d-model) (alength ^doubles actual)))
      (is (arr-approx= expected actual 1.0e-6)
          (str "packed batched heads preserve the headwise result at d-model=" d-model)))))

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

(deftest graph-attention-generic-scatter-algebra-matches-sequential-oracle
  (let [n-nodes 3
        n-edges 4
        d-model 2
        h (double-array [0.2 -0.3, 0.5 0.7, -0.4 0.9])
        Wq (double-array [0.8 -0.1, 0.3 0.6])
        Wk (double-array [0.4 0.2, -0.5 0.9])
        Wv (double-array [0.7 -0.2, 0.1 0.5])
        Wo (double-array [0.6 0.4, -0.3 0.8])
        src (long-array [0 1 2 0])
        dst (long-array [2 2 0 1])
        mm (fn [^doubles a ^doubles b]
             (let [out (double-array (* n-nodes d-model))]
               (dotimes [i n-nodes]
                 (dotimes [j d-model]
                   (aset out (+ (* i d-model) j)
                         (reduce +
                                 (for [k (range d-model)]
                                   (* (aget a (+ (* i d-model) k))
                                      (aget b (+ (* k d-model) j))))))))
               out))
        q (mm h Wq)
        k (mm h Wk)
        v (mm h Wv)
        scores (double-array n-edges)
        denominator (double-array n-nodes)
        weighted (double-array (* n-nodes d-model))
        scale (/ 1.0 (Math/sqrt (double d-model)))]
    (dotimes [edge n-edges]
      (let [source (aget src edge)
            destination (aget dst edge)
            dot (reduce +
                        (for [component (range d-model)]
                          (* (aget q (+ (* destination d-model) component))
                             (aget k (+ (* source d-model) component)))))
            score (Math/exp (min 5.0 (max -5.0 (* dot scale))))]
        (aset scores edge score)
        (aset denominator destination (+ (aget denominator destination) score))
        (dotimes [component d-model]
          (let [to (+ (* destination d-model) component)
                from (+ (* source d-model) component)]
            (aset weighted to (+ (aget weighted to) (* score (aget v from))))))))
    (dotimes [node n-nodes]
      (dotimes [component d-model]
        (let [index (+ (* node d-model) component)]
          (aset weighted index
                (/ (aget weighted index) (+ (aget denominator node) 1e-6))))))
    (let [expected (mm weighted Wo)
          actual (attn/graph-attention h Wq Wk Wv Wo src dst
                                       n-nodes n-edges d-model)]
      (is (arr-approx= expected actual 1e-12)))))

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

(deftest rope-prefill-source-shaped-test
  (let [rows 3 heads 2 head-dim 6 half (quot head-dim 2)
        x (float-array (map #(float (/ (- % 13) 9.0))
                            (range (* rows heads head-dim))))
        actual (float-array (alength x))
        expected (float-array (alength x))
        theta 10000.0]
    (attn/rope-prefill! x actual rows heads head-dim theta)
    (dotimes [row rows]
      (dotimes [head heads]
        (let [base (+ (* row heads head-dim) (* head head-dim))]
          (dotimes [i half]
            (let [angle (* row (Math/pow theta (/ (* -2.0 i) head-dim)))
                  c (Math/cos angle) s (Math/sin angle)
                  x0 (aget x (+ base i)) x1 (aget x (+ base i half))]
              (aset expected (+ base i) (float (- (* x0 c) (* x1 s))))
              (aset expected (+ base i half) (float (+ (* x1 c) (* x0 s)))))))))
    (dotimes [i (alength x)]
      (is (< (Math/abs (- (double (aget expected i)) (double (aget actual i)))) 1e-6)))))

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
    (testing "value-returning softmax matches the in-place compatibility primitive"
      (let [source (scores attn/attn-prefill-scores-bidir!)
            expected (aclone source)
            actual (attn/attn-prefill-softmax source T heads)]
        (attn/attn-prefill-softmax! expected T heads)
        (dotimes [i (alength expected)]
          (is (< (Math/abs (- (double (aget expected i)) (double (aget actual i)))) 1e-6)))))
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

(deftest rope-pos-rows-buffered-test
  (let [nrows 3 heads 2 head-dim 8 theta 10000.0
        positions (int-array [0 7 31])
        x (float-array (map #(float (/ (- % 17) 13.0))
                            (range (* nrows heads head-dim))))
        out (float-array (alength x))]
    (attn/rope-pos-rows-buf! x out nrows heads head-dim theta positions)
    (dotimes [row nrows]
      (let [offset (* row heads head-dim)
            input-row (float-array (* heads head-dim))]
        (System/arraycopy x offset input-row 0 (alength input-row))
        (let [expected (attn/rope-pos input-row 1 heads head-dim theta (aget positions row))]
          (dotimes [i (alength input-row)]
            (is (< (Math/abs (- (double (aget ^floats expected i))
                                (double (aget out (+ offset i)))))
                   1.0e-5)
                (str "row " row ", element " i))))))
    (let [in-place (aclone x)]
      (attn/rope-pos-rows-buf! in-place in-place nrows heads head-dim theta positions)
      (is (java.util.Arrays/equals out ^floats in-place)
          "each work-item reads its pair before an in-place write"))))

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
            base ^floats (attention-reference/gqa-decode q kc vc n n-q n-kv hd scale)
            got ^floats (attn/gqa-decode-attention-weights! q kc vc n n-q n-kv hd scale wsink)]
        (testing (str "output bit-identical to the sequential oracle (n-q " n-q " n-kv " n-kv ")")
          (is (java.util.Arrays/equals base got)))
        (testing "head-averaged weights sum to ~1 for the query"
          (let [s (loop [j 0 s 0.0] (if (< j n) (recur (inc j) (+ s (aget wsink j))) s))]
            (is (approx= s 1.0 1e-5))))
        (testing "wsink ACCUMULATES across calls (per-layer averaging contract)"
          (attn/gqa-decode-attention-weights! q kc vc n n-q n-kv hd scale wsink)
          (let [s (loop [j 0 s 0.0] (if (< j n) (recur (inc j) (+ s (aget wsink j))) s))]
            (is (approx= s 2.0 1e-5))))))))
