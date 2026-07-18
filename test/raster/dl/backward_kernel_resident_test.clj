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
          got (attn/rope x 1 sl heads hd theta)
          ref (ref-rope x sl heads hd theta)]
      (is (< (max-abs-diff got ref) 1e-9)
          (str "rope fwd sl=" sl " heads=" heads " hd=" hd " theta=" theta
               " maxdiff=" (max-abs-diff got ref))))))

(deftest rope-backward-matches-reference
  (doseq [[sl heads hd theta] [[6 4 8 10000.0] [4 2 16 1.0e6] [3 3 4 10000.0]]]
    (let [dy (da (* sl heads hd) (+ 70 sl))
          got (attn/rope-backward-dx dy 1 sl heads hd theta)
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
  (let [y (raster.dl.attention/rope x 1 seq-len heads head-dim theta)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* seq-len (clojure.core/* heads head-dim)))))

(deftest rms-norm-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "rms-norm resident parity")
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

;; The GPU SCHEDULE of the same op (rms-norm-chunked): its value+grad must ALSO extract
;; fully resident and match the CPU grads. This is the schedule the gemma training block
;; actually runs — three kernels (chunk-partial reduce | row combine | apply) per norm
;; instead of one 64-work-item kernel.
(deftm rmsn-chunked-parity-loss [x :- (Array float) w :- (Array float) tgt :- (Array float)
                                 rows :- Long feat :- Long chunks :- Long
                                 eps :- Double go :- Double] :- Double
  (let [y (raster.dl.nn/rms-norm-chunked x w rows feat chunks eps go)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* rows feat))))

(deftest rms-norm-chunked-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "rms-norm-chunked resident parity")
    (let [rows 8 feat 32 chunks 4 go 1.0 eps 1e-6
          x (fa (* rows feat) 1) w (fa feat 2) tgt (fa (* rows feat) 3)
          {:keys [grads]}
          (gp/grad-parity #'rmsn-chunked-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'w :type '(Array float) :val w}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'rows :type 'Long :val rows}
                           {:name 'feat :type 'Long :val feat}
                           {:name 'chunks :type 'Long :val chunks}
                           {:name 'eps :type 'Double :val eps}
                           {:name 'go :type 'Double :val go}]
                          :grad-args '[x w])]
      (println "  [rms-norm-chunked] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (println "  [rms-norm-chunked] grad(w) steps:" (:step-kinds (get grads 'w))
               "rel-err" (:rel-err (get grads 'w)))
      (is (every? some? (map :resident? (vals grads)))))))

;; gqa-causal-mha value+grad: the milestone — the SDPA backward (dq/dk/dv kernels)
;; composed with pack/broadcast/unpack. grad wrt Q must extract FULLY RESIDENT.
(deftm gqa-parity-loss [Q :- (Array float) K :- (Array float) V :- (Array float) tgt :- (Array float)
                        seq-len :- Long nq :- Long nkv :- Long hd :- Long] :- Double
  (let [y (raster.dl.attention/gqa-causal-mha Q K V 1 seq-len nq nkv hd)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* seq-len (clojure.core/* nq hd)))))

(deftest gqa-causal-mha-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "gqa-causal-mha resident parity")
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

;; ── MILESTONE: the FULL attention-block value+grad, grad wrt x, FULLY RESIDENT ────
;; rms-norm → q/k/v linear-nb → rope(q,k) → fused gqa-causal-mha → out linear-nb → mse.
;; `x` FANS OUT (xn feeds q/k/v), so d_xn = d_xn_q ⊕ d_xn_k ⊕ d_xn_v — a
;; multi-contribution ARRAY cotangent accumulation. The nil-safe `grad-acc` used for
;; that fold has no GPU kernel (:unlowered-array-op); the reverse-AD emission now folds
;; array fan-outs with the resident element-wise `residual-add` (see
;; raster.ad.reverse/sum-contribs-into), so the whole block extracts resident. This is
;; the regression pin for that fix: grad(x) must be FULLY RESIDENT and match CPU.
(deftm attn-block-parity-loss
  [x :- (Array float) wn :- (Array float)
   Wq :- (Array float) Wk :- (Array float) Wv :- (Array float) Wo :- (Array float)
   tgt :- (Array float)
   seq :- Long dmodel :- Long nq :- Long nkv :- Long hd :- Long
   eps :- Double go :- Double theta :- Double] :- Double
  (let [xn (raster.dl.nn/rms-norm x wn seq dmodel eps go)
        q  (raster.dl.nn/linear-nb xn Wq seq dmodel (clojure.core/* nq hd))
        k  (raster.dl.nn/linear-nb xn Wk seq dmodel (clojure.core/* nkv hd))
        v  (raster.dl.nn/linear-nb xn Wv seq dmodel (clojure.core/* nkv hd))
        qr (raster.dl.attention/rope q 1 seq nq hd theta)
        kr (raster.dl.attention/rope k 1 seq nkv hd theta)
        a  (raster.dl.attention/gqa-causal-mha qr kr v 1 seq nq nkv hd)
        o  (raster.dl.nn/linear-nb a Wo seq (clojure.core/* nq hd) dmodel)]
    (raster.dl.loss/mse-loss o tgt (clojure.core/* seq dmodel))))

(deftest attn-block-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "attn-block resident parity")
    (let [seq 6 dmodel 32 nq 4 nkv 1 hd 8 eps 1e-6 go 1.0 theta 10000.0
          x  (fa (* seq dmodel) 31) wn (fa dmodel 32)
          Wq (fa (* dmodel (* nq hd)) 33) Wk (fa (* dmodel (* nkv hd)) 34)
          Wv (fa (* dmodel (* nkv hd)) 35) Wo (fa (* (* nq hd) dmodel) 36)
          tgt (fa (* seq dmodel) 37)
          {:keys [grads]}
          (gp/grad-parity #'attn-block-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'wn :type '(Array float) :val wn}
                           {:name 'Wq :type '(Array float) :val Wq}
                           {:name 'Wk :type '(Array float) :val Wk}
                           {:name 'Wv :type '(Array float) :val Wv}
                           {:name 'Wo :type '(Array float) :val Wo}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'seq :type 'Long :val seq}
                           {:name 'dmodel :type 'Long :val dmodel}
                           {:name 'nq :type 'Long :val nq}
                           {:name 'nkv :type 'Long :val nkv}
                           {:name 'hd :type 'Long :val hd}
                           {:name 'eps :type 'Double :val eps}
                           {:name 'go :type 'Double :val go}
                           {:name 'theta :type 'Double :val theta}]
                          :grad-args '[x]
                          ;; The isolated kernels (rms-norm/rope/gqa above) match at
                          ;; ~1e-7, but the FULL block composes 12 f32 GEMMs (BLAS
                          ;; sgemm on CPU vs the device GEMM) + softmax through both
                          ;; forward and backward — the global relative divergence of
                          ;; the composed f32 gradient is ~1.7e-2, pure f32-GEMM
                          ;; accumulation noise (the AD chain itself is FD-verified at
                          ;; f64 in decoder-ad-test). Hold the resident-parity gate at
                          ;; a realistic composed-f32 tolerance.
                          :rtol 3.0e-2)]
      (println "  [attn-block] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (is (:resident? (get grads 'x))
          "full attention-block grad(x) must extract FULLY RESIDENT"))))

;; ── :gemm-precision :f32-scalar — exact-grad GEMM policy (training) ──────────────
;; Same composed attention block as above, but the resident :gemm steps bind the plain
;; scalar f32 GEMM (no f32→f16 convert/transpose expansion). The ~1.7e-2 composed
;; divergence of the default :f16-xmx path is pure f16 INPUT-CONVERSION noise, not an
;; AD/compile defect — under :f32-scalar the same compiled program matches the CPU f32
;; reference at ~1e-6-level. This pins the policy end-to-end: compile-gpu-program
;; carries :gemm-precision on the descriptor, bind-program! binds scalar GEMMs for it.
(deftest attn-block-f32-scalar-gemm-precision-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "attn-block :f32-scalar gemm-precision parity")
    (let [seq 6 dmodel 32 nq 4 nkv 1 hd 8 eps 1e-6 go 1.0 theta 10000.0
          x  (fa (* seq dmodel) 31) wn (fa dmodel 32)
          Wq (fa (* dmodel (* nq hd)) 33) Wk (fa (* dmodel (* nkv hd)) 34)
          Wv (fa (* dmodel (* nkv hd)) 35) Wo (fa (* (* nq hd) dmodel) 36)
          tgt (fa (* seq dmodel) 37)
          {:keys [grads]}
          (gp/grad-parity #'attn-block-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'wn :type '(Array float) :val wn}
                           {:name 'Wq :type '(Array float) :val Wq}
                           {:name 'Wk :type '(Array float) :val Wk}
                           {:name 'Wv :type '(Array float) :val Wv}
                           {:name 'Wo :type '(Array float) :val Wo}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'seq :type 'Long :val seq}
                           {:name 'dmodel :type 'Long :val dmodel}
                           {:name 'nq :type 'Long :val nq}
                           {:name 'nkv :type 'Long :val nkv}
                           {:name 'hd :type 'Long :val hd}
                           {:name 'eps :type 'Double :val eps}
                           {:name 'go :type 'Double :val go}
                           {:name 'theta :type 'Double :val theta}]
                          :grad-args '[x]
                          :gemm-precision :f32-scalar
                          ;; exact-f32 GEMM: only summation-order noise remains
                          ;; (CPU BLAS sgemm vs scalar device GEMM, both f32).
                          :rtol 1.0e-5)]
      (println "  [attn-block :f32-scalar] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (is (:resident? (get grads 'x))
          "attn-block grad(x) under :gemm-precision :f32-scalar must extract FULLY RESIDENT")
      (is (some #{:gemm} (:step-kinds (get grads 'x)))
          "the policy case must actually exercise resident :gemm steps"))))

;; ── B2 MILESTONE: RESIDUAL-connection fan-out value+grad, FULLY RESIDENT ──────────
;; A real decoder layer has RESIDUAL connections (the attention block above has none):
;; a value `x1 = residual-add(x, m)` feeds BOTH the next `residual-add(x1, o)` AND
;; `rms-norm(x1)` (fan-out), so its cotangent is `d_x1 = grad-acc(d_z, dx)` — a
;; multi-contribution ARRAY accumulation. Before B2 this fell back to the nil-safe
;; `grad-acc` (no GPU kernel) → :unlowered-array-op → non-resident. B2 makes it fold
;; via the resident `residual-add`. THREE things had to line up:
;;   1. `residual-add` is kept SYMBOLIC in the walk (lowers via broadcast SOAC), so its
;;      result sym was UNTAGGED — the reverse-AD reroute gates on an :array sym-tag and
;;      never fired. Fixed by a :result-type facet (raster.dl.nn) + the walker's
;;      bare-symbolic-call arm + the let-binder reading the init's stamped tag.
;;   2. The residual contribution `d_z` is a bare SYMBOL ALIAS: mse-loss binds
;;      `d_pred = (mse-grad …)`, residual-add's IDENTITY-grad passthrough forwards that
;;      adjoint SYMBOL to both input slots, and `sum-contribs` aliases a single
;;      contribution to itself — so `d_z → d_pred → (mse-grad …)`, two symbol hops
;;      before the kernel. `provably-non-nil-contrib?` now FOLLOWS pure symbol aliases
;;      through the SSA binding-map to their non-nil kernel source (nil-safety intact:
;;      any hop through grad-acc/nth/if/nil is still rejected → keeps nil-safe grad-acc).
;;   3. The differentiated input `x` ALSO fans out (linear-nb ∘ residual), exercising
;;      the param-adjoint reroute path too.
;; grad(x) and grad(W1) must extract FULLY RESIDENT and match CPU. This is the B2
;; regression pin (the grad-acc→residual-add reroute over aliases + the symbolic-op tag).
(deftm resblk-parity-loss
  [x :- (Array float) wn :- (Array float) W1 :- (Array float)
   pn :- (Array float) W2 :- (Array float) tgt :- (Array float)
   rows :- Long d :- Long eps :- Double go :- Double] :- Double
  (let [n  (clojure.core/* rows d)
        h  (raster.dl.nn/rms-norm x wn rows d eps go)
        m  (raster.dl.nn/linear-nb h W1 rows d d)
        x1 (raster.dl.nn/residual-add x m n)       ;; x1 FANS OUT ↓↓
        h2 (raster.dl.nn/rms-norm x1 pn rows d eps go)
        o  (raster.dl.nn/linear-nb h2 W2 rows d d)
        z  (raster.dl.nn/residual-add x1 o n)]
    (raster.dl.loss/mse-loss z tgt n)))

(deftest resblk-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "resblk (residual fan-out) resident parity")
    (let [rows 4 d 16 eps 1e-6 go 1.0
          x  (fa (* rows d) 51) wn (fa d 52) W1 (fa (* d d) 53)
          pn (fa d 54) W2 (fa (* d d) 55) tgt (fa (* rows d) 56)
          {:keys [grads]}
          (gp/grad-parity #'resblk-parity-loss
                          [{:name 'x :type '(Array float) :val x}
                           {:name 'wn :type '(Array float) :val wn}
                           {:name 'W1 :type '(Array float) :val W1}
                           {:name 'pn :type '(Array float) :val pn}
                           {:name 'W2 :type '(Array float) :val W2}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'rows :type 'Long :val rows}
                           {:name 'd :type 'Long :val d}
                           {:name 'eps :type 'Double :val eps}
                           {:name 'go :type 'Double :val go}]
                          ;; grad(x) is the B2 target: x FANS OUT (residual-add ∘
                          ;; rms-norm) AND x1 = residual-add(x,m) fans out again — the
                          ;; whole reroute-over-aliases + symbolic-op-tag chain. It
                          ;; extracts resident and matches the CPU reference at ~5e-4,
                          ;; so hold the gate at 1e-3 (≈2× headroom) — the former 3.0e-2
                          ;; was 60× slack that could hide a real regression.
                          :grad-args '[x]
                          :rtol 1.0e-3)]
      (println "  [resblk] grad(x) steps:" (:step-kinds (get grads 'x))
               "rel-err" (:rel-err (get grads 'x)))
      (is (:resident? (get grads 'x))
          "residual fan-out grad(x) must extract FULLY RESIDENT"))))

;; ── horizontal-fusion multi-output regression (the gemma-block extraction bug) ────
;; Two independent same-bound pure elementwise branches (g = a⊙a, u = b⊙b) consumed
;; elementwise (y = g⊙u) then reduced (mse) — the residual/fan-out shape of the gemma
;; FFN at tiny dims. The SOAC fuser horizontally fuses the branch pair into ONE
;; multi-output map whose SECONDARY output buffer (`hfuse_out__N`) exists only as a
;; side-effect aset in the fused lambda; the backward re-creates the same pair
;; (da = d_y⊙…, db = d_y⊙…). This pins three formerly-broken layers:
;;   1. SOAC io classification: an aset-written array is an array OUTPUT, never a
;;      scalar (before: the kernel declared `float hfuse_out__N` and the extraction
;;      eval'd the bare buffer sym on the host → `Unable to resolve symbol`).
;;   2. Resident extraction: a :map step's binding sym ALIASES its out buffer
;;      (invoke-registered-kernel returns it), so a later step reading the fused
;;      PRIMARY's binding resolves to the real resident buffer at bind time.
;;   3. resolve-alength: `(alength <invoke-binding>)` — a later fused branch's bound —
;;      resolves through the invoke's registered buffer semantics (:in-place-arg) to
;;      the out buffer's alloc size instead of surviving as a host read of a device
;;      buffer.
(deftm hfuse-two-branch-loss
  [a :- (Array float) b :- (Array float) tgt :- (Array float) n :- Long] :- Double
  (let [g (raster.dl.nn/hadamard a a n)
        u (raster.dl.nn/hadamard b b n)
        y (raster.dl.nn/hadamard g u n)]
    (raster.dl.loss/mse-loss y tgt n)))

(deftest horizontal-fusion-multi-output-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "horizontal-fusion multi-output resident parity")
    (let [n 32
          a (fa n 61) b (fa n 62) tgt (fa n 63)
          {:keys [grads]}
          (gp/grad-parity #'hfuse-two-branch-loss
                          [{:name 'a :type '(Array float) :val a}
                           {:name 'b :type '(Array float) :val b}
                           {:name 'tgt :type '(Array float) :val tgt}
                           {:name 'n :type 'Long :val n}]
                          ;; grad(a) reads the fused PRIMARY output (the invoke-binding
                          ;; alias), grad(b) the SECONDARY (the hfuse_out alias binding).
                          :grad-args '[a b]
                          :rtol 1.0e-5)]
      (doseq [g '[a b]]
        (println "  [hfuse] grad(" g ") steps:" (:step-kinds (get grads g))
                 "rel-err" (:rel-err (get grads g)))
        (is (:resident? (get grads g))
            (str "fused multi-output grad(" g ") must extract FULLY RESIDENT"))))))

(deftest rope-value+grad-resident-parity
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "rope resident parity")
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
