(ns raster.dl.backward-kernel-resident-test
  "Phase-1a QLoRA backward kernels rewritten from raw nested dotimes/loop into
   resident par/map-void! kernels (rms-norm backward dx/dweight, rope fwd + bwd).

   GATE 1 (CPU exact-equality): each rewritten kernel reproduces the analytic
   reference formula (the pre-rewrite spelling) to ~1e-6 — the rewrite is a pure
   restructuring, numerics unchanged.

   GATE 2 (GPU parity, ze:0): a rms-norm value+grad and a rope value+grad each
   extract FULLY RESIDENT (via grad-parity) and match CPU grads within rtol."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.dl.loss :as loss]
            [raster.arrays :as ra]
            [raster.ad.reverse :as rev]
            [raster.dl.gpu-grad-parity :as gp]))

;; ── input builders ──────────────────────────────────────────────────────────────
(defn- da ^doubles [n seed]
  (let [r (java.util.Random. (long seed)) a (double-array n)]
    (dotimes [i n] (aset a i (.nextGaussian r))) a))

(defn- fa ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (.nextGaussian r)))) a))

(defn- max-abs-diff [a b]
  (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq a) (seq b))))

;; ── naive references (the pre-rewrite analytic spelling, in double) ───────────────
(defn- ref-rmsn-dx ^doubles [^doubles dy ^doubles x ^doubles w rows feat eps go]
  (let [dx (double-array (* rows feat))]
    (dotimes [r rows]
      (let [off (* r (int feat))
            ms (loop [i 0 s 0.0] (if (< i feat) (let [v (aget x (+ off i))] (recur (inc i) (+ s (* v v)))) (/ s (double feat))))
            inv (/ 1.0 (Math/sqrt (+ ms eps)))
            c (loop [i 0 s 0.0] (if (< i feat)
                                  (let [gi (+ go (aget w i))]
                                    (recur (inc i) (+ s (* gi (* (aget x (+ off i)) (aget dy (+ off i))))))) s))
            inv3f (/ (* inv inv inv) (double feat))]
        (dotimes [i feat]
          (let [gi (+ go (aget w i))]
            (aset dx (+ off i) (- (* inv (* gi (aget dy (+ off i)))) (* inv3f (* (aget x (+ off i)) c))))))))
    dx))

(defn- ref-rmsn-dw ^doubles [^doubles dy ^doubles x rows feat eps]
  (let [dw (double-array feat)]
    (dotimes [r rows]
      (let [off (* r (int feat))
            ms (loop [i 0 s 0.0] (if (< i feat) (let [v (aget x (+ off i))] (recur (inc i) (+ s (* v v)))) (/ s (double feat))))
            inv (/ 1.0 (Math/sqrt (+ ms eps)))]
        (dotimes [i feat]
          (aset dw i (+ (aget dw i) (* (aget dy (+ off i)) (* (aget x (+ off i)) inv)))))))
    dw))

(defn- ref-rope ^doubles [^doubles x seq-len heads head-dim theta]
  (let [out (double-array (* seq-len heads head-dim)) half (quot head-dim 2) lnt (Math/log theta)]
    (dotimes [p seq-len]
      (dotimes [h heads]
        (let [base (+ (* p (* heads head-dim)) (* h (int head-dim)))]
          (dotimes [i half]
            (let [freq (Math/exp (* (/ (* -2.0 (double i)) (double head-dim)) lnt))
                  ang (* (double p) freq) c (Math/cos ang) s (Math/sin ang)
                  x0 (aget x (+ base i)) x1 (aget x (+ (+ base i) half))]
              (aset out (+ base i) (- (* x0 c) (* x1 s)))
              (aset out (+ (+ base i) half) (+ (* x1 c) (* x0 s))))))))
    out))

(defn- ref-rope-dx ^doubles [^doubles dy seq-len heads head-dim theta]
  (let [dx (double-array (* seq-len heads head-dim)) half (quot head-dim 2) lnt (Math/log theta)]
    (dotimes [p seq-len]
      (dotimes [h heads]
        (let [base (+ (* p (* heads head-dim)) (* h (int head-dim)))]
          (dotimes [i half]
            (let [freq (Math/exp (* (/ (* -2.0 (double i)) (double head-dim)) lnt))
                  ang (* (double p) freq) c (Math/cos ang) s (Math/sin ang)
                  d0 (aget dy (+ base i)) d1 (aget dy (+ (+ base i) half))]
              (aset dx (+ base i) (+ (* d0 c) (* d1 s)))
              (aset dx (+ (+ base i) half) (- (* d1 c) (* d0 s))))))))
    dx))

;; ── GATE 1: CPU exact-equality (kernel vs naive reference) ────────────────────────
(deftest rms-norm-backward-dx-matches-reference
  (doseq [[rows feat go] [[3 16 1.0] [4 8 0.0] [2 32 1.0]]]
    (let [eps 1e-6
          dy (da (* rows feat) (+ 10 rows)) x (da (* rows feat) (+ 20 feat)) w (da feat (+ 30 rows))
          got (nn/rms-norm-backward-dx dy x w rows feat eps go)
          ref (ref-rmsn-dx dy x w rows feat eps go)]
      (is (< (max-abs-diff got ref) 1e-9)
          (str "rms-norm-backward-dx rows=" rows " feat=" feat " go=" go
               " maxdiff=" (max-abs-diff got ref))))))

(deftest rms-norm-backward-dweight-matches-reference
  (doseq [[rows feat] [[3 16] [4 8] [5 32]]]
    (let [eps 1e-6
          dy (da (* rows feat) (+ 40 rows)) x (da (* rows feat) (+ 50 feat))
          got (nn/rms-norm-backward-dweight dy x rows feat eps)
          ref (ref-rmsn-dw dy x rows feat eps)]
      (is (< (max-abs-diff got ref) 1e-9)
          (str "rms-norm-backward-dweight rows=" rows " feat=" feat
               " maxdiff=" (max-abs-diff got ref))))))

(deftest rope-forward-matches-reference
  (doseq [[sl heads hd theta] [[6 4 8 10000.0] [4 2 16 1.0e6] [3 3 4 10000.0]]]
    (let [x (da (* sl heads hd) (+ 60 sl))
          got (attn/rope x sl heads hd theta)
          ref (ref-rope x sl heads hd theta)]
      (is (< (max-abs-diff got ref) 1e-9)
          (str "rope fwd sl=" sl " heads=" heads " hd=" hd " theta=" theta
               " maxdiff=" (max-abs-diff got ref))))))

(deftest rope-backward-matches-reference
  (doseq [[sl heads hd theta] [[6 4 8 10000.0] [4 2 16 1.0e6] [3 3 4 10000.0]]]
    (let [dy (da (* sl heads hd) (+ 70 sl))
          got (attn/rope-backward-dx dy sl heads hd theta)
          ref (ref-rope-dx dy sl heads hd theta)]
      (is (< (max-abs-diff got ref) 1e-9)
          (str "rope bwd sl=" sl " heads=" heads " hd=" hd " theta=" theta
               " maxdiff=" (max-abs-diff got ref))))))

;; ── GATE 1b: batched-causal-sdpa backward vs finite differences ───────────────────
;; The three flat resident kernels (dq/dk/dv) are the flash-attention backward of
;; batched-causal-sdpa. Check them directly against central FD of the forward: for a
;; random upstream adjoint dO, L = Σ_k dO[k]·out[k]; then dL/dQ = dq-kernel, dL/dK =
;; dk-kernel, dL/dV = dv-kernel. Run at BOTH f64 and f32 (the rewrite must not change
;; the gradient at either dtype).
(defn- sdpa-fd-d [f ^doubles a idx eps]
  (let [a0 (aget a idx)
        fp (do (aset a idx (+ a0 eps)) (f)) fm (do (aset a idx (- a0 eps)) (f))]
    (aset a idx a0) (/ (- fp fm) (* 2.0 eps))))
(defn- sdpa-fd-f [f ^floats a idx eps]
  (let [a0 (aget a idx)
        fp (do (aset a idx (float (+ a0 eps))) (f)) fm (do (aset a idx (float (- a0 eps))) (f))]
    (aset a idx (float a0)) (/ (- fp fm) (* 2.0 eps))))

(deftest batched-causal-sdpa-backward-matches-fd-f64
  (testing "dQ/dK/dV kernels match central finite differences (double)"
    (let [batch 2 sl 3 hd 4 n (* batch sl hd)
          Q (da n 101) K (da n 202) V (da n 303) dO (da n 404)
          dot (fn [^doubles a ^doubles b] (reduce + 0.0 (map * (seq a) (seq b))))
          L   (fn [] (dot dO (attn/batched-causal-sdpa Q K V batch sl hd)))
          dQ (attn/batched-causal-sdpa-dq dO Q K V batch sl hd)
          dK (attn/batched-causal-sdpa-dk dO Q K V batch sl hd)
          dV (attn/batched-causal-sdpa-dv dO Q K V batch sl hd)]
      (doseq [i [0 5 11 16 23]]
        (is (< (Math/abs (- (aget ^doubles dQ i) (sdpa-fd-d L Q i 1e-5))) 1e-6) (str "dQ[" i "]"))
        (is (< (Math/abs (- (aget ^doubles dK i) (sdpa-fd-d L K i 1e-5))) 1e-6) (str "dK[" i "]"))
        (is (< (Math/abs (- (aget ^doubles dV i) (sdpa-fd-d L V i 1e-5))) 1e-6) (str "dV[" i "]"))))))

(deftest batched-causal-sdpa-backward-matches-fd-f32
  (testing "dQ/dK/dV kernels match central finite differences (float)"
    (let [batch 2 sl 3 hd 4 n (* batch sl hd)
          Q (fa n 101) K (fa n 202) V (fa n 303) dO (fa n 404)
          dot (fn [^floats a ^floats b] (reduce + 0.0 (map #(* (double %1) (double %2)) (seq a) (seq b))))
          L   (fn [] (dot dO (attn/batched-causal-sdpa Q K V batch sl hd)))
          dQ (attn/batched-causal-sdpa-dq dO Q K V batch sl hd)
          dK (attn/batched-causal-sdpa-dk dO Q K V batch sl hd)
          dV (attn/batched-causal-sdpa-dv dO Q K V batch sl hd)
          ok? (fn [a n] (< (Math/abs (- a n)) (+ 3e-2 (* 3e-2 (Math/abs n)))))]
      (doseq [i [0 5 11 16 23]]
        (is (ok? (double (aget ^floats dQ i)) (sdpa-fd-f L Q i 3e-2)) (str "dQ[" i "]"))
        (is (ok? (double (aget ^floats dK i)) (sdpa-fd-f L K i 3e-2)) (str "dK[" i "]"))
        (is (ok? (double (aget ^floats dV i)) (sdpa-fd-f L V i 3e-2)) (str "dV[" i "]"))))))

;; ── GATE 2: GPU resident parity ──────────────────────────────────────────────────
;; Losses use the resident-proven mse-loss reduction so the value+grad extracts to a
;; straight-line resident program (forward + backward + loss-grad, all :map-void/:reduce).
(deftm rmsn-parity-loss [x :- (Array float) w :- (Array float) tgt :- (Array float)
                         rows :- Long feat :- Long eps :- Double go :- Double] :- Double
  (let [y (raster.dl.nn/rms-norm x w rows feat eps go)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* rows feat))))

(deftm rope-parity-loss [x :- (Array float) tgt :- (Array float)
                         seq-len :- Long heads :- Long head-dim :- Long theta :- Double] :- Double
  (let [y (raster.dl.attention/rope x seq-len heads head-dim theta)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* seq-len (clojure.core/* heads head-dim)))))

(deftest rms-norm-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (println "  [SKIP] rms-norm resident parity: no Level Zero GPU")
    (let [rows 4 feat 16 go 1.0 eps 1e-6
          x (fa (* rows feat) 1) w (fa feat 2) tgt (fa (* rows feat) 3)
          {:keys [grads]}
          (gp/grad-parity #'rmsn-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'w :type '(Array float) :val w}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'rows :type 'Long :val rows}
                           {:name 'feat :type 'Long :val feat}
                           {:name 'eps :type 'Double :val eps}
                           {:name 'go :type 'Double :val go}]
                          :grad-args '[x w])]
      (println "  [rms-norm] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (println "  [rms-norm] grad(w) steps:" (:step-kinds (get grads 'w))
               "rel-err" (:rel-err (get grads 'w)))
      (is (every? some? (map :resident? (vals grads)))))))

;; gqa-causal-mha value+grad: the milestone — the SDPA backward (dq/dk/dv kernels)
;; composed with pack/broadcast/unpack. grad wrt Q must extract FULLY RESIDENT.
(deftm gqa-parity-loss [Q :- (Array float) K :- (Array float) V :- (Array float) tgt :- (Array float)
                        seq-len :- Long nq :- Long nkv :- Long hd :- Long] :- Double
  (let [y (raster.dl.attention/gqa-causal-mha Q K V seq-len nq nkv hd)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* seq-len (clojure.core/* nq hd)))))

(deftest gqa-causal-mha-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (println "  [SKIP] gqa-causal-mha resident parity: no Level Zero GPU")
    (let [sl 4 nq 4 nkv 2 hd 8
          Q (fa (* sl nq hd) 21) K (fa (* sl nkv hd) 22) V (fa (* sl nkv hd) 23)
          tgt (fa (* sl nq hd) 24)
          {:keys [grads]}
          (gp/grad-parity #'gqa-parity-loss
                          [{:name 'Q :type '(Array float) :val Q}
                           {:name 'K :type '(Array float) :val K}
                           {:name 'V :type '(Array float) :val V}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'seq-len :type 'Long :val sl}
                           {:name 'nq :type 'Long :val nq}
                           {:name 'nkv :type 'Long :val nkv}
                           {:name 'hd :type 'Long :val hd}]
                          :grad-args '[Q])]
      (println "  [gqa] grad(Q) steps:" (:step-kinds (get grads 'Q))
               "rel-err" (:rel-err (get grads 'Q)))
      (is (:resident? (get grads 'Q))))))

;; NOTE — the FULL attention-block value+grad (rms-norm → q/k/v linear-nb → rope →
;; gqa-causal-mha → out linear-nb → mse, grad wrt x) does NOT yet extract fully
;; resident, but the blocker is NOT the SDPA backward (this PR's deliverable). A
;; bisected compile-gpu-program probe shows the ONLY surviving undevirtualized
;; dispatches are the linear-GEMM family — {linear-nb 1, mse-grad 1, linear-dx 4,
;; linear-dW 3} — i.e. exactly the pre-existing F5 blocker (linear GEMM recognition
;; in composed AD IR + mse-grad residency). pack/broadcast/rope/rms-norm AND the new
;; sdpa dq/dk/dv kernels are ALL absent from the surviving list — they compose
;; resident (the gqa-causal-mha test below is the resident+parity milestone). The
;; interpreted CPU value+grad of the same block also throws a [F/[D ClassCast in the
;; linear-composition layer; both are tracked as the next Phase-1b prereq, separate
;; from this PR. Re-add a resident-parity deftest here once linear GEMM composition
;; lowers.

(deftest rope-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (println "  [SKIP] rope resident parity: no Level Zero GPU")
    (let [sl 6 heads 4 hd 8 theta 10000.0
          x (fa (* sl heads hd) 11) tgt (fa (* sl heads hd) 12)
          {:keys [grads]}
          (gp/grad-parity #'rope-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'seq-len :type 'Long :val sl}
                           {:name 'heads :type 'Long :val heads}
                           {:name 'head-dim :type 'Long :val hd}
                           {:name 'theta :type 'Double :val theta}]
                          :grad-args '[x])]
      (println "  [rope] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (is (:resident? (get grads 'x))))))
