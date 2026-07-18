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
