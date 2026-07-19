(ns raster.compiler.core.hardware-roofline-test
  "Phase-0.1: the analytic roofline reads per-dtype peak-flops + bandwidth off the descriptor
   (XLA gpu_performance_model shape). Device-free — a hand-built descriptor stands in for a probe."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hw]))

;; An Arc-140V-shaped descriptor (f32 ~4 TF, f64 1/16, ~90 GB/s) with the perf fields the probe
;; would supply. f16 intentionally omitted to exercise the f32 fallback.
(def ^:private desc
  {:device-type :gpu :device-id :ze:0
   :bandwidth-bytes-s 89.6e9
   :peak-flops {:f32 3.9936e12 :float 3.9936e12 :f64 0.2496e12 :double 0.2496e12}
   :balance 60})

(def ^:private no-perf-desc
  {:device-type :gpu :device-id :ze:0 :balance 60})   ;; nothing probed/measured yet

(deftest peak-flops-lookup
  (is (= 3.9936e12 (hw/peak-flops-for desc :f32)))
  (is (= 0.2496e12 (hw/peak-flops-for desc :f64)))
  (testing "f16 falls back to the f32 ceiling when no separate half-rate is reported"
    (is (= 3.9936e12 (hw/peak-flops-for desc :f16))))
  (is (nil? (hw/peak-flops-for no-perf-desc :f32))))

(deftest balance-is-precision-dependent
  (testing "ridge = peak-flops(dtype)/bandwidth — f32 ridge ≫ f64 ridge on a mixed-rate GPU"
    (is (< 44.0 (hw/balance-for desc :f32) 45.0) "f32 ridge ~44.6")
    (is (< 2.7 (hw/balance-for desc :f64) 2.9) "f64 ridge ~2.79")
    (is (> (hw/balance-for desc :f32) (* 10 (hw/balance-for desc :f64)))))
  (testing "falls back to the scalar :balance default when bandwidth/flops are absent"
    (is (= 60.0 (hw/balance-for no-perf-desc :f32)))))

(deftest regime-fixes-the-single-balance-bug
  ;; a 256³ GEMM: AI = 2·256³ / (3·256²·4) = 42.67 — BETWEEN the f64 ridge (2.8) and f32 ridge (44.6)
  (let [k {:flops (* 2 256 256 256) :bytes (* 3 256 256 4)}]
    (testing "the SAME kernel is compute-bound at f64 but memory-bound at f32 (precision-aware)"
      (is (= :compute-bound (hw/roofline-regime desc k :f64)))
      (is (= :memory-bound (hw/roofline-regime desc k :f32))))
    (testing "the 2-arg legacy form still uses the scalar :balance (back-compat)"
      (is (= :memory-bound (hw/roofline-regime desc k))))))   ;; 42.67 < 60

(deftest roofline-time-and-the-fusion-lever
  (let [k {:flops (* 2 1024 1024 1024) :bytes (* 3 1024 1024 4) :dtype :f32}]
    (testing "returns an ns estimate when the descriptor carries bandwidth+flops"
      (is (pos? (hw/roofline-time-ns desc k))))
    (testing "nil (not a fabricated number) when the descriptor has no perf fields yet"
      (is (nil? (hw/roofline-time-ns no-perf-desc k))))
    (testing "each extra kernel charges the 1µs launch tax — the analytic fusion lever"
      (let [t1 (hw/roofline-time-ns desc (assoc k :n-kernels 1))
            t8 (hw/roofline-time-ns desc (assoc k :n-kernels 8))]
        (is (< 6990.0 (- t8 t1) 7010.0) "8 vs 1 kernels differ by ~7×1µs = 7000 ns")))
    (testing "a supplied measured launch overhead overrides the 1µs default"
      (let [t (hw/roofline-time-ns desc (assoc k :n-kernels 2 :launch-overhead-ns 50.0))
            t0 (hw/roofline-time-ns desc (assoc k :n-kernels 0))]
        (is (< 99.0 (- t t0) 101.0) "2×50ns measured tax")))))

;; --- B0a: how the roofline must be APPLIED to price a fusion (validated vs measurement) ---
;; A training linear A[M×640]·B[640×2048] → C, then a bias add. We MEASURED epilogue fusion
;; (fold the bias into the GEMM store, one kernel) vs unfused (two kernels, C round-trips):
;;   M=1 → +0.8%, M=128 → −3.2%, M=512 → +14.6%, M=1024 → +17.7% (device-event median, this session).
;; Two lessons the pricer (B3) MUST encode, both asserted here:
;;  (1) Price a fusion as the SUM OF PER-KERNEL roofline times — NOT sum-the-traffic-then-overlap-once.
;;      The naive single-overlap form overlaps the *separate* bias kernel's memory with the GEMM's
;;      compute, which inverts the trend (predicts fusion helps MOST at M=1, least at M=1024 — the
;;      opposite of measured). Per-kernel pricing recovers the correct trend: tiny win at M=1, large
;;      win at big M.
;;  (2) The roofline is a directional SEED, never the arbiter: even priced per-kernel it predicts
;;      ~+22% at M=128 where measurement showed a −3.2% REGRESSION. That sign flip is an occupancy
;;      effect the FLOP+byte+launch model is structurally blind to → measurement gates the decision.
(deftest roofline-fusion-pricing-is-per-kernel
  (let [N 2048 K 640
        gemm-flops (fn [M] (* 2 M K N))
        q-gemm (fn [M] (+ (* M K 2) (* K N 2) (* M N 4)))        ;; A(f16)+B(f16)+Cwrite(f32)
        q-bias (fn [M] (+ (* M N 4) (* M N 4) (* N 4)))          ;; Cread+Cwrite+bias
        naive%   (fn [M]                                          ;; WRONG: one overlap over summed traffic
                   (let [u (hw/roofline-time-ns desc {:flops (gemm-flops M)
                                                      :bytes (+ (q-gemm M) (q-bias M)) :dtype :f32 :n-kernels 2})
                         f (hw/roofline-time-ns desc {:flops (gemm-flops M)
                                                      :bytes (+ (q-gemm M) (* N 4)) :dtype :f32 :n-kernels 1})]
                     (* 100.0 (/ (- u f) u))))
        perk%    (fn [M]                                          ;; RIGHT: sum of per-kernel times
                   (let [u (+ (hw/roofline-time-ns desc {:flops (gemm-flops M) :bytes (q-gemm M) :dtype :f32 :n-kernels 1})
                              (hw/roofline-time-ns desc {:flops (* M N) :bytes (q-bias M) :dtype :f32 :n-kernels 1}))
                         f (hw/roofline-time-ns desc {:flops (gemm-flops M)
                                                      :bytes (+ (q-gemm M) (* N 4)) :dtype :f32 :n-kernels 1})]
                     (* 100.0 (/ (- u f) u))))]
    (testing "naive summed-then-overlapped pricing INVERTS the trend (documents the trap)"
      (is (> (naive% 1) (naive% 1024))
          "the wrong model predicts fusion helps more at M=1 than M=1024 — opposite of measured"))
    (testing "per-kernel pricing recovers the measured trend: small win at M=1, large at big M"
      (is (< (perk% 1) 5.0)   "M=1: near-neutral (measured +0.8%)")
      (is (> (perk% 512) 15.0) "M=512: substantial (measured +14.6%)")
      (is (< (perk% 1) (perk% 512)) "monotone up with M, unlike the naive model"))
    (testing "roofline is only a SEED — it cannot see the M=128 occupancy regression"
      (is (pos? (perk% 128))
          "roofline predicts a WIN at M=128, but measurement showed −3.2% → measurement must gate"))))
