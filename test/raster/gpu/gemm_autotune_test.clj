(ns raster.gpu.gemm-autotune-test
  "The autotune loop end-to-end on REAL kernels: coordinate-descent (raster.gpu.autotune) driven by a
   real device-event GEMM benchmark finds the measured-optimal split-k factor on a deep-k,
   occupancy-bound shape — the axis with the largest proven win (~10×). This is the proof that the
   Phase-2 machinery tunes actual kernels, not a synthetic cost.

   Honesty: split-k is a wired, high-headroom axis (occupancy). The dominant proj GEMM on this Arc
   iGPU is already ~ceiling and its inner-loop levers (grf256/prefetch/dbuf) measured DEAD; the
   small-m attention slab's gap to oneDNN is DECOMPOSITION (M=64 underfills the XMX 30×), not a
   tuning knob. So the autotuner earns its keep on the occupancy axis (split-k), where measurement —
   not a 'maximize splits' heuristic — finds the true optimum (which is interior, not the extreme)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.core.hardware :as hw]
            [raster.gpu.autotune :as at]))

;; ── the real cost-fn: device-event time of the resident split-k XMX GEMM ─────────
(defn- bench-splitk-ms
  "Median device-event ms of the split-k GEMM (+reduce) at (m,n,k) with `splits` k-chunks."
  [ze m n k splits]
  (let [g!   (ns-resolve ze 'make-buffer)
        f16  (ns-resolve ze 'buffer-of-floats-as-half)
        rec  (ns-resolve ze 'record-graph!)   rep (ns-resolve ze 'replay-graph!)
        rst  (ns-resolve ze 'reset-graph-events!) rts (ns-resolve ze 'read-graph-timestamps!)
        dst  (ns-resolve ze 'destroy-graph!)
        skg  (ns-resolve ze 'bind-registered-gemm-splitk!)
        skr  (ns-resolve ze 'bind-registered-splitk-reduce!)
        free (ns-resolve ze 'free-buffer!)
        rng  (java.util.Random. 3)
        mk   (fn [s] (let [a (float-array s)] (dotimes [i s] (aset a i (float (.nextGaussian rng)))) a))
        kc   (* 32 (quot (quot (long k) (long splits)) 32))
        a16  (f16 (mk (* m k)))  b16 (f16 (mk (* k n)))
        part (g! (* splits m n) :float)  c (g! (* m n) :float)]
    (try
      (let [g (rec [{:bound (skg a16 b16 part m n k kc splits) :kernel-name "gemm_nonsquare_splitk"}
                    {:bound (skr part c (* m n) splits) :kernel-name "splitk_reduce"}]
                   {:profile? true})]
        (try
          (dotimes [_ 3] (rep g) (rst g))
          (let [s (vec (for [_ (range 11)] (do (rep g) (reduce + (map :ms (:kernels (rts g)))))))]
            (nth (sort s) 5))
          (finally (dst g))))
      (finally (free a16) (free b16) (free part) (free c)))))

(deftest autotune-finds-splitk-optimum
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "GEMM autotune: split-k optimum on the resident XMX GEMM")
    (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          m 16 n 512 k 131072                     ;; deep-k, small (m,n) grid → occupancy-bound
          gflops (fn [ms] (/ (* 2.0 m k n) (* ms 1.0e6)))
          measured (atom 0)
          ;; the REAL cost the loop minimises: measured device ms of the actual kernel
          cost (fn [cfg] (swap! measured inc) (bench-splitk-ms ze m n k (:splits cfg)))
          baseline-ms (bench-splitk-ms ze m n k 1)
          result (at/coordinate-descent {:splits 1}
                                        {:splits (fn [s] [(* s 2) (max 1 (quot s 2))])}
                                        cost)
          tuned (:splits (:config result))
          tuned-ms (:cost result)
          speedup (/ baseline-ms tuned-ms)]
      (println (format (str "\n=== GEMM split-k autotune (m=%d n=%d k=%d) ===\n"
                            "  baseline splits=1 : %.2f GFLOPS\n"
                            "  autotuned splits=%d: %.2f GFLOPS  (%.1fx, %d kernels measured)\n")
                       m n k (gflops baseline-ms) tuned (gflops tuned-ms) speedup @measured))
      (testing "coordinate-descent measures real kernels and finds a large occupancy win"
        (is (>= tuned 8) "the optimum is a high split factor (deep-k underfills a single grid)")
        (is (> speedup 4.0) (str "tuned split-k must beat the non-split baseline by >4× (got "
                                 (format "%.1fx" speedup) ")"))
        (is (<= @measured 12) "the ×2/÷2 ladder from the seed keeps the search cheap")))))

;; ── T3: the TILE axis — every candidate tile is correct; coordinate-descent prices real kernels ──
(defn- run-tiled-gemm
  "Run bind-registered-gemm-tiled! for `tile` at (m,n,k) over fixed A/B, returning [C median-ms]."
  [ze a16 b16 m n k tile]
  (let [g!   (ns-resolve ze 'make-buffer)   free (ns-resolve ze 'free-buffer!)
        rec  (ns-resolve ze 'record-graph!) rep (ns-resolve ze 'replay-graph!)
        rst  (ns-resolve ze 'reset-graph-events!) rts (ns-resolve ze 'read-graph-timestamps!)
        dst  (ns-resolve ze 'destroy-graph!) ba (ns-resolve ze 'buffer->array)
        btg  (ns-resolve ze 'bind-registered-gemm-tiled!)
        c    (g! (* m n) :float)]
    (try
      (let [g (rec [{:bound (btg a16 b16 c m n k :float tile) :kernel-name "t"}] {:profile? true})]
        (try
          (dotimes [_ 2] (rep g) (rst g))
          (let [ms (vec (for [_ (range 7)] (do (rep g) (reduce + (map :ms (:kernels (rts g)))))))]
            [(vec (ba c)) (nth (sort ms) 3)])
          (finally (dst g))))
      (finally (free c)))))

(deftest autotune-tile-axis-all-candidates-correct
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "GEMM autotune: tile axis (all candidates correct + measured search)")
    (let [ze   (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          f16  (ns-resolve ze 'buffer-of-floats-as-half)  free (ns-resolve ze 'free-buffer!)
          desc (hw/descriptor-for :ze:0)
          cands (hw/gemm-tile-candidates desc)
          default-tile (hw/derive-gemm-tile desc)
          m 256 n 256 k 512
          rng (java.util.Random. 4)
          mk  (fn [s] (let [a (float-array s)] (dotimes [i s] (aset a i (float (* 0.05 (.nextGaussian rng))))) a))
          a16 (f16 (mk (* m k)))  b16 (f16 (mk (* k n)))]
      (try
        (testing "the curated candidate list is non-trivial and includes the derived default"
          (is (> (count cands) 1) "more than one tile to search")
          (is (some #(= % default-tile) cands) "the derived default is in the search space"))
        (let [ref (first (run-tiled-gemm ze a16 b16 m n k default-tile))
              results (mapv (fn [t] (let [[c ms] (run-tiled-gemm ze a16 b16 m n k t)]
                                      {:tile t :bit-identical? (= ref c) :ms ms})) cands)]
          (testing "SAFETY: every candidate tile produces output bit-identical to the default"
            (doseq [{:keys [tile bit-identical?]} results]
              (is bit-identical? (str "tile " (select-keys tile [:block-m :block-n :block-k]) " must match default"))))
          ;; coordinate-descent over the tile axis, priced by real device time
          (let [cost (fn [cfg] (:ms (first (filter #(= (:tile %) (:tile cfg)) results))))
                r (at/coordinate-descent {:tile default-tile}
                                         {:tile (fn [_] cands)} cost)
                winner (:tile (:config r))]
            (println (format "\n=== GEMM tile autotune (m=%d n=%d k=%d) ===" m n k))
            (doseq [{:keys [tile ms]} (sort-by :ms results)]
              (println (format "  %-22s %.3f ms%s" (str (:block-m tile) "x" (:block-n tile) " k" (:block-k tile))
                               ms (if (= tile winner) "  <- winner" ""))))
            (testing "measured search returns a feasible candidate (Arc: hand-tuned default is ~optimal)"
              (is (some #(= % winner) cands) "winner is one of the candidates")
              (is (= ref (first (run-tiled-gemm ze a16 b16 m n k winner))) "winner output is correct"))))
        (finally (free a16) (free b16))))))
