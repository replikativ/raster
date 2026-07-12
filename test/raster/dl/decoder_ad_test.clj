(ns raster.dl.decoder-ad-test
  "AD rules for the decoder-LM ops that LoRA fine-tuning of Gemma/Qwen needs
   beyond rms-norm/linear-nb (see norm-ad-test): rope (rotation pullback),
   hadamard / swiglu (gated MLP), and GQA/MQA causal attention. Each analytic
   gradient is checked against central finite differences."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.dl.loss :as loss]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.ad.reverse :as rev]))

(defn- faf ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (.nextGaussian r)))) a))
(defn- fad ^doubles [n seed]
  (let [r (java.util.Random. (long seed)) a (double-array n)]
    (dotimes [i n] (aset a i (* 0.3 (.nextGaussian r)))) a))
(defn- fdf [f ^floats a idx eps]
  (let [a0 (aget a idx)
        fp (do (aset a idx (float (+ a0 eps))) (f)) fm (do (aset a idx (float (- a0 eps))) (f))]
    (aset a idx (float a0)) (/ (- fp fm) (* 2.0 eps))))
(defn- fdd [f ^doubles a idx eps]
  (let [a0 (aget a idx)
        fp (do (aset a idx (+ a0 eps)) (f)) fm (do (aset a idx (- a0 eps)) (f))]
    (aset a idx a0) (/ (- fp fm) (* 2.0 eps))))

;; ---- RoPE ----
(deftm rope-loss [x :- (Array float) tgt :- (Array float)
                  seq :- Long heads :- Long hd :- Long theta :- Double] :- Double
  (let [y (raster.dl.attention/rope x 1 seq heads hd theta)
        n* (clojure.core/* seq (clojure.core/* heads hd))]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e))))) s))))

(deftest rope-grad
  (testing "RoPE dx matches finite differences (rotation pullback)"
    (let [seq 4 heads 2 hd 8 theta 10000.0
          x (faf (* seq heads hd) 1) tgt (faf (* seq heads hd) 2)
          gx (nth ((rev/value+grad #'rope-loss) x tgt seq heads hd theta) 1)
          f (fn [] (rope-loss x tgt seq heads hd theta))]
      (doseq [i [0 4 9 40 63]]
        (is (< (Math/abs (- (double (aget ^floats gx i)) (fdf f x i 1e-3))) 2e-3)
            (str "rope dx[" i "]"))))))

;; ---- SwiGLU (uses hadamard + silu + linear-nb) ----
(deftm swiglu-loss [x :- (Array float) G :- (Array float) U :- (Array float) D :- (Array float)
                    tgt :- (Array float) rows :- Long d :- Long ff :- Long] :- Double
  (let [y (raster.dl.nn/swiglu x G U D rows d ff) n* (clojure.core/* rows d)]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e))))) s))))

(deftest swiglu-grad
  (testing "SwiGLU gate/up/down gradients match finite differences"
    (let [rows 2 d 8 ff 16
          x (faf (* rows d) 1) G (faf (* ff d) 2) U (faf (* ff d) 3) D (faf (* d ff) 4)
          tgt (faf (* rows d) 5)
          res ((rev/value+grad #'swiglu-loss) x G U D tgt rows d ff)
          gG (nth res 2) gU (nth res 3) gD (nth res 4)
          f (fn [] (swiglu-loss x G U D tgt rows d ff))]
      ;; relative tolerance: swiglu gradients are large-magnitude (unnormalized
      ;; sum loss), so absolute 3e-3 would reject finite-diff truncation noise
      (letfn [(ok? [a n] (< (Math/abs (- a n)) (+ 5e-3 (* 5e-3 (Math/abs n)))))]
        (doseq [i [0 5 40]] (is (ok? (double (aget ^floats gG i)) (fdf f G i 1e-3)) (str "dG[" i "]")))
        (doseq [i [0 20]]   (is (ok? (double (aget ^floats gU i)) (fdf f U i 1e-3)) (str "dU[" i "]")))
        (doseq [i [0 30]]   (is (ok? (double (aget ^floats gD i)) (fdf f D i 1e-3)) (str "dD[" i "]")))))))

;; ---- GQA / MQA causal attention ----
(deftm gqa-loss [Q :- (Array double) K :- (Array double) V :- (Array double) tgt :- (Array double)
                 seq :- Long nq :- Long nkv :- Long hd :- Long] :- Double
  (raster.dl.loss/mse-loss (raster.dl.attention/gqa-causal-mha Q K V 1 seq nq nkv hd)
                           tgt (clojure.core/* seq (clojure.core/* nq hd))))

(deftest gqa-attention-grad
  (testing "GQA causal attention dQ/dK/dV match finite differences (head-sharing)"
    (let [seq 4 nq 4 nkv 2 hd 8
          Q (fad (* seq nq hd) 1) K (fad (* seq nkv hd) 2) V (fad (* seq nkv hd) 3)
          tgt (fad (* seq nq hd) 4)
          res ((rev/value+grad #'gqa-loss) Q K V tgt seq nq nkv hd)
          gQ (nth res 1) gK (nth res 2) gV (nth res 3)
          f (fn [] (gqa-loss Q K V tgt seq nq nkv hd))]
      ;; dQ at later positions (position 0 has a single causal key → dQ=0 there)
      (doseq [i [33 40 70 127]]
        (is (< (Math/abs (- (aget ^doubles gQ i) (fdd f Q i 1e-4))) 1e-5) (str "dQ[" i "]")))
      (doseq [i [0 20 40]] (is (< (Math/abs (- (aget ^doubles gK i) (fdd f K i 1e-4))) 1e-5) (str "dK[" i "]")))
      (doseq [i [0 20 40]] (is (< (Math/abs (- (aget ^doubles gV i) (fdd f V i 1e-4))) 1e-5) (str "dV[" i "]"))))))

(deftest mqa-attention-grad
  (testing "MQA (n_kv=1) is the GQA edge case and differentiates"
    (let [seq 3 nq 4 nkv 1 hd 8
          Q (fad (* seq nq hd) 6) K (fad (* seq nkv hd) 7) V (fad (* seq nkv hd) 8)
          tgt (fad (* seq nq hd) 9)
          res ((rev/value+grad #'gqa-loss) Q K V tgt seq nq nkv hd)
          gK (nth res 2)
          f (fn [] (gqa-loss Q K V tgt seq nq nkv hd))]
      (doseq [i [0 5 15]]
        (is (< (Math/abs (- (aget ^doubles gK i) (fdd f K i 1e-4))) 1e-5) (str "dK[" i "]"))))))
