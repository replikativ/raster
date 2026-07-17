(ns raster.dl.dv-batched-gemm-spike
  "SPIKE #2 (branch gpu/batched-gemm): attention backward dV as ONE resident BATCHED
   XMX GEMM over the heads, not a loop of single-head GEMMs.

   dV[b] = W[b]ᵀ · dO[b] per [seq,hd] slab (W carries the causal structural zeros,
   so a dense transpose-A GEMM is the correct masked result). The FIRST spike
   (spike/dv-as-gemm) got NO-GO because a per-slab GEMM at gemma's seq=64 is m=k=64
   — ~30× too small to fill the XMX array — and a resident LOOP of GEMMs is not even
   expressible. THIS spike batches all 64 slabs into one launch over a 3D grid
   (z = slab), so the DPAS array is fed across ALL slabs: 64 slabs × (1×2) per-slab
   tiles = 128 workgroups, which FILL the ~32-workgroup iGPU where one slab (2 wg)
   starves it.

   Kernel: raster.gpu.ze-runtime/bind-registered-gemm-batched! (emit-gemm-nonsquare-kernel
   :batched? true). This is the batched NN primitive C[b]=A[b]·B[b]; the -tn layout
   (Wᵀ) is staged host-side here (transpose each W slab → A) to isolate the GEMM lever
   the GATE measures.

   GATE (real shape 64 slabs, m=k=64, n=256): device-event GFLOPS ≥ ~900 (½ of the
   oneDNN 1829-GFLOPS ceiling, ~10× the 79-GFLOPS scalar path) → the lever is real."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]))

(def ^:private gpu-available?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- fa ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (* 0.5 (.nextGaussian r))))) a))
(defn- da ^doubles [n seed]
  (let [r (java.util.Random. (long seed)) a (double-array n)]
    (dotimes [i n] (aset a i (.nextGaussian r))) a))

(defn- median [xs]
  (let [v (vec (sort xs)) n (count v)]
    (if (odd? n) (nth v (quot n 2))
        (/ (+ (nth v (dec (quot n 2))) (nth v (quot n 2))) 2.0))))

;; ── Host transpose-A staging: A[b] = W[b]ᵀ, flattened [batch, seq, seq] ───────────
(defn- transpose-slabs ^floats [^floats W batch seq-len]
  (let [ss (* seq-len seq-len) out (float-array (* batch ss))]
    (dotimes [b batch]
      (let [o (* b ss)]
        (dotimes [i seq-len]
          (dotimes [j seq-len]
            ;; A[b][j,i] = W[b][i,j]
            (aset out (+ o (* j seq-len) i) (aget W (+ o (* i seq-len) j)))))))
    out))

;; ── ALGEBRAIC correctness (f64, machine precision): dV = Wᵀ·dO per slab ───────────
(deftest dv-batched-gemm-algebra-f64
  (let [batch 8 sl 12 hd 16 n (* batch sl hd)
        dO (da n 404) Q (da n 101) K (da n 202) V (da n 303)
        scalar (attn/batched-causal-sdpa-dv dO Q K V batch sl hd)
        W (attn/batched-causal-attn-weights Q K batch sl hd)
        ss (* sl sl) slab (* sl hd)
        ;; per-slab dgemm-tn! (= nn/linear-dW): dV[j,d]=Σ_i W[i,j]·dO[i,d]
        gemm (let [out (double-array n)]
               (dotimes [b batch]
                 (let [Wsl (java.util.Arrays/copyOfRange ^doubles W (* b ss) (* (inc b) ss))
                       dOsl (java.util.Arrays/copyOfRange ^doubles dO (* b slab) (* (inc b) slab))
                       dVsl ^doubles (nn/linear-dW Wsl dOsl sl hd sl)]
                   (System/arraycopy dVsl 0 out (* b slab) slab)))
               out)
        num (reduce max 0.0 (map (fn [a b] (Math/abs (- a b))) scalar gemm))
        den (reduce max 1e-12 (map #(Math/abs %) scalar))]
    (is (< (/ num den) 1e-12)
        (str "dV batched-GEMM algebra vs scalar rel-err " (/ num den)))))

;; ── GPU: resident batched XMX GEMM at the REAL shape; correctness + GFLOPS gate ───
(deftest dv-batched-gemm-gpu-gate
  (if-not @gpu-available?
    (println "  [SKIP] dv batched-gemm gate: no Level Zero GPU")
    (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          make-buffer (ns-resolve ze 'make-buffer)
          buf-f16-of  (ns-resolve ze 'buffer-of-floats-as-half)
          upload!     (ns-resolve ze 'array->buffer!)
          download    (ns-resolve ze 'buffer->array)
          free!       (ns-resolve ze 'free-buffer!)
          record!     (ns-resolve ze 'record-graph!)
          replay!     (ns-resolve ze 'replay-graph!)
          reset-ev!   (ns-resolve ze 'reset-graph-events!)
          read-ts!    (ns-resolve ze 'read-graph-timestamps!)
          destroy!    (ns-resolve ze 'destroy-graph!)
          batched!    (ns-resolve ze 'bind-registered-gemm-batched!)
          ;; real gemma attention shape: 64 slabs (B·n-heads), seq=64, hd=256
          batch 64 sl 64 hd 256
          m sl k sl n hd
          nQ (* batch sl hd)
          Q (fa nQ 11) K (fa nQ 22) dO (fa nQ 33) V (fa nQ 44)
          scalar ^floats (attn/batched-causal-sdpa-dv dO Q K V batch sl hd)
          W ^floats (attn/batched-causal-attn-weights Q K batch sl hd)
          A (transpose-slabs W batch sl)           ;; [batch, m=seq, k=seq]  (= Wᵀ per slab)
          B dO                                     ;; [batch, k=seq, n=hd]
          a16 (buf-f16-of A)
          b16 (buf-f16-of B)
          c   (make-buffer nQ :float)]
      (try
        (let [g (record! [{:bound (batched! a16 b16 c m n k batch)
                           :kernel-name "gemm_nonsquare_batched"}]
                         {:profile? true})]
          (try
            ;; warmup
            (dotimes [_ 3] (replay! g) (reset-ev! g))
            ;; timed replays (interleaved medians)
            (let [samples (vec (for [_ (range 13)]
                                 (do (replay! g)
                                     (let [ts (read-ts! g)]
                                       (-> ts :kernels first :ms)))))
                  med-ms (median samples)
                  flops (* 2.0 batch m k n)
                  gflops (/ flops (* med-ms 1.0e6))
                  ceil 1829.0 scalar-gflops 79.0
                  out ^floats (download c)
                  numr (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) out scalar))
                  denr (reduce max 1e-9 (map #(Math/abs (double %)) scalar))
                  relerr (/ numr denr)]
              (println (format (str "\n=== dV batched-GEMM GATE (batch=%d m=%d k=%d n=%d) ===\n"
                                    "  samples-ms (median of 13): %.4f  [min %.4f max %.4f]\n"
                                    "  GFLOPS: %.1f   %% of oneDNN 1829 ceiling: %.1f%%\n"
                                    "  vs scalar 79 GFLOPS: %.1fx\n"
                                    "  rel-err (f16 XMX) vs scalar dV: %.2e\n"
                                    "  GATE >=900 GFLOPS: %s")
                               batch m k n med-ms (reduce min samples) (reduce max samples)
                               gflops (* 100.0 (/ gflops ceil)) (/ gflops scalar-gflops)
                               relerr (if (>= gflops 900.0) "PASS" "FAIL")))
              (is (< relerr 5.0e-3)
                  (str "batched-GEMM dV vs scalar rel-err " relerr " (f16 floor ~1e-3)")))
            (finally (destroy! g))))
        (finally (doseq [b [a16 b16 c]] (free! b)))))))
