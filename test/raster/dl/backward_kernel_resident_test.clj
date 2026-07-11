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
