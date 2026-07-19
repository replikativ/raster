(ns raster.perf.regression-gate-test
  "B0.5 — the perf-regression RATCHET. The one gate that catches wall-time
   regressions the structural proxies (buffer counts, split-k policy, tile-fits-budget)
   cannot see: a GEMM-leaf rewrite, epilogue fusion, or cost-guided fusion can keep
   every proxy identical while regressing µs. This closes that blind spot.

   HOW IT RUNS. ^:perf tests are EXCLUDED from CircleCI (JDK-24, no Valhalla / no GPU
   → not a representative machine) via the :test alias's `-e :perf`. Run locally on the
   Valhalla+Arc box:
     clojure -M:perf                    ;; gate against the committed baseline
     clojure -M:perf --update-baseline  ;; re-seed after an INTENDED speedup (ratchet down)

   HOW IT GATES. Each canary times a compiled kernel with microbench/do-bench (device
   kernels will use profile-program! as GPU canaries land — B1) and asserts its
   STATIONARY median has not regressed past baseline × SLACK. The slack band is wide
   enough that the ±2–3% steady-state noise never flakes, tight enough to catch a real
   ≥15% regression. A missing baseline for this machine RECORDS instead of failing (first
   run seeds it); the baseline is keyed by a machine signature so numbers never cross
   machines. Only stationary samples are compared — a high-CV read retries rather than
   flaking. This turns the existing (shape-only) measurement layer into an actual gate
   with no new measurement code.

   COVERAGE (grows with the agenda — each canary guards the phase that could regress it):
     - cpu/simd-reduce      : device-free, proves the mechanism + guards the JVM-SIMD path.
     - TODO B1: gemm-leaf   : fixed-shape GEMM device median (profile-program!).
     - TODO B2: epilogue    : fused-epilogue microkernel host median.
     - TODO A3: train-step  : MLP/LeNet compiled train-step µs/step."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [raster.runtime.microbench :as mb]))

(def ^:private SLACK
  "Regression tolerance: median ≤ baseline × SLACK passes. 1.15 clears the ±2–3%
   steady-state noise (characterized in bench/resident_gemm_cold_bench.clj and the
   finetune sft benchmark) with margin, while still catching a real ≥15% regression."
  1.15)

(def ^:private baseline-file (io/file "test" "resources" "perf_baseline.edn"))

(defn- machine-sig
  "Perf identity of THIS machine for device-free canaries — cores × JVM build. GPU
   canaries (B1) extend this with autotune/descriptor-signature so a device swap or a
   re-calibration gets a fresh baseline rather than a false regression."
  []
  (str (.availableProcessors (Runtime/getRuntime)) "c-"
       (System/getProperty "java.vm.version")))

(defn- load-baseline []
  (if (.exists baseline-file) (edn/read-string (slurp baseline-file)) {}))

(defn- update-baseline? []
  (boolean (some #{"--update-baseline"} (or (seq *command-line-args*) []))))

(defn gate!
  "Time `thunk` with do-bench and gate its stationary median against the committed
   baseline for (machine, canary-key). Seeds the baseline when absent or when
   --update-baseline is passed; otherwise asserts median ≤ baseline × SLACK. Skips the
   comparison (and retries via a failed stationarity assert) if the sample is not
   stationary, so noise never masquerades as a regression."
  [canary-key thunk]
  ;; warmup-ms 500: let C2 fully compile before measuring (the default 25ms leaves the
  ;; kernel interpreted → false non-stationarity). cv-threshold 0.08: the median is
  ;; rock-stable across budgets; only the spread flirts with a tighter bound, and a
  ;; high-CV sample retries rather than gating, so 0.08 trades no regression-detection
  ;; for far fewer stationarity flakes on a lightly-loaded box.
  (let [{:keys [median-ns stationary?]} (mb/do-bench thunk :warmup-ms 500 :budget-ms 800 :cv-threshold 0.08)
        sig  (machine-sig)
        base (load-baseline)
        prev (get-in base [sig canary-key])]
    (is stationary?
        (str canary-key ": measurement not stationary (CV over threshold) — rerun"))
    (when stationary?
      (if (or (nil? prev) (update-baseline?))
        (do (io/make-parents baseline-file)
            (spit baseline-file (pr-str (assoc-in base [sig canary-key] (long median-ns))))
            (is true (str canary-key ": baseline "
                          (if prev "re-seeded" "seeded") " @ " (long median-ns)
                          "ns (" sig ")")))
        (is (<= (double median-ns) (* SLACK (double prev)))
            (str canary-key ": " (long median-ns) "ns exceeds baseline " (long prev)
                 "ns × " SLACK " — regressed "
                 (format "%.1f%%" (* 100.0 (dec (/ (double median-ns) (double prev)))))))))))

;; --- Canary 1 (device-free): a C2-vectorizable double reduction ---
;; Guards the JVM-SIMD lowering path and proves the ratchet end-to-end without a device.
;; A tight sum-of-products over a fixed array — the shape C2 SuperWord vectorizes; a
;; regression here (e.g. a boxing or a lost FloatVector selection) shows as µs growth.
(deftest ^:perf cpu-simd-reduce-canary
  (testing "JVM-SIMD reduction canary within regression slack"
    (let [n (bit-shift-left 1 20)                 ;; ~1M elems → ms-scale, above timer noise (stationary)
          a (double-array n)
          _ (dotimes [i n] (aset a i (double (* 0.5 (mod i 97)))))]
      (gate! :cpu/simd-reduce
             (fn [] (let [^doubles a a]
                      (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (aget a i) (aget a i)))) s))))))))
