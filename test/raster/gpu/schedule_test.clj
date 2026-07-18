(ns raster.gpu.schedule-test
  "S6 PR-1 acceptance: schedule-as-data + the register-budget feasibility gate.

   T1–T3 are device-free (the schedule is pure data+fns). T4 checks the schedule threads
   through compile-gpu-program and the gate fires on the real Arc."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.gpu.schedule :as sched]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.pipeline :as pl]))

;; A hand-built Arc-140V-shaped descriptor (grf128, subgroup 16 → 256 B/lane) — lets the pure
;; schedule fns be tested with no live device.
(def ^:private arc-desc
  {:device-type :gpu :device-id :ze:0
   :machine-lanes 8192 :subgroup-size 16 :grf-bytes-per-lane 256
   :max-workgroup-size 1024})

;; ════════════════════════════════════════════════════════════════════════════════
;; T1 — parity: :gemm-precision sugar ≡ :schedule {:precision …} (no device)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest t1-precision-parity
  (let [derived (sched/derive-default nil arc-desc)]
    (testing "the deprecated :gemm-precision sugar resolves to the same :precision as an explicit override"
      (is (= :f32-scalar (:precision (sched/resolve derived {:gemm-precision :f32-scalar}))))
      (is (= :f32-scalar (:precision (sched/resolve derived {:precision :f32-scalar}))))
      (is (= (sched/resolve derived {:gemm-precision :f32-scalar})
             (sched/resolve derived {:precision :f32-scalar}))
          "sugar and explicit override produce byte-identical schedules"))
    (testing "the default policy is f16-xmx (training AMP)"
      (is (= :f16-xmx (:precision derived))))
    (testing "a pinned override is recorded in :meta :overrides"
      (is (contains? (get-in (sched/resolve derived {:precision :f32-scalar}) [:meta :overrides])
                     :precision)))))

;; ════════════════════════════════════════════════════════════════════════════════
;; T2 — the gate: rejects register double-buffering, passes the default (no device)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest t2-feasibility-gate
  (testing "the DEFAULT f16-xmx GEMM (stage :none) is feasible at grf128 — 256 ≤ 256"
    (is (true? (sched/feasible? (sched/derive-default nil arc-desc) arc-desc))))
  (testing "f32-scalar (quarter-file accumulator) is comfortably feasible"
    (is (true? (sched/feasible? (sched/resolve (sched/derive-default nil arc-desc)
                                               {:precision :f32-scalar})
                                arc-desc))))
  (testing "register double-buffering BOTH operands on f16-xmx at grf128 is REJECTED before emit"
    (let [infeasible (sched/resolve (sched/derive-default nil arc-desc)
                                    {:stage {:space :register :copies {:a 2 :b 2}}})
          ex (try (sched/feasible? infeasible arc-desc) nil
                  (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (some? ex) "must throw an ex-info, not silently pass")
      (is (= 512 (:required-bytes-per-lane ex)) "256 acc + 4×64 staged = 512")
      (is (= 256 (:budget-bytes-per-lane ex)))
      (is (> (:required-bytes-per-lane ex) (:budget-bytes-per-lane ex)))))
  (testing "SLM/L3 staging is FREE against the GRF budget (the winning staging space)"
    (is (true? (sched/feasible? (sched/resolve (sched/derive-default nil arc-desc)
                                               {:stage {:space :slm :copies {:a 2 :b 2}}})
                                arc-desc))))
  (testing "grf256 DOUBLES the budget (opt-in, for the deep-K follow-up)"
    (let [s (sched/resolve (sched/derive-default nil arc-desc)
                           {:grf {:mode :grf256} :stage {:space :register :copies {:a 1 :b 0}}})]
      ;; f16-xmx acc scales with the file (512), + 1×128 staged = 640 > 512 → still rejected,
      ;; but a f32-scalar acc (128) + 128 staged = 256 ≤ 512 fits.
      (is (true? (sched/feasible? (assoc s :precision :f32-scalar) arc-desc))))))

;; ════════════════════════════════════════════════════════════════════════════════
;; T3 — derive-default: no dead knobs on by default (no device)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest t3-derived-default
  (let [d (sched/derive-default nil arc-desc)]
    (testing "precision default is f16-xmx"
      (is (= :f16-xmx (:precision d))))
    (testing "the dead inner-loop knobs are OFF by default"
      (is (= :grf128 (get-in d [:grf :mode])))
      (is (= :none (get-in d [:stage :space]))))
    (testing ":meta carries the machine params the gate reads"
      (is (= 256 (get-in d [:meta :machine-params :grf-bytes-per-lane])))
      (is (= 16 (get-in d [:meta :machine-params :subgroup])))
      (is (= :ze:0 (get-in d [:meta :target]))))))

;; ════════════════════════════════════════════════════════════════════════════════
;; T4 — the schedule threads through compile-gpu-program + the gate fires (device)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest t4-schedule-flows-through-compile
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "S6 schedule threads through compile-gpu-program")
    (let [hw (requiring-resolve 'raster.compiler.core.hardware/descriptor-for)
          desc (hw :ze:0)]
      (testing "descriptor-for exposes the real Arc GRF budget"
        (is (= 256 (:grf-bytes-per-lane desc))
            "Arc 140V grf128, subgroup 16 → 256 B/lane")
        (is (= 16 (:subgroup-size desc))))
      (require 'raster.dl.gemma-train-resident-test)
      (let [step-var (ns-resolve 'raster.dl.gemma-train-resident-test 'gblk-train-step)]
        (testing ":gemm-precision sugar and :schedule override both resolve on the descriptor"
          (let [via-sugar (pl/compile-gpu-program step-var :ze:0 :dtype :float
                                                  :on-non-resident :nil :gemm-precision :f32-scalar)
                via-sched (pl/compile-gpu-program step-var :ze:0 :dtype :float
                                                  :on-non-resident :nil
                                                  :schedule {:precision :f32-scalar})]
            (is (= :f32-scalar (get-in via-sugar [:schedule :precision])))
            (is (= :f32-scalar (get-in via-sched [:schedule :precision])))
            (is (= :f32-scalar (:gemm-precision via-sugar))
                "back-compat :gemm-precision mirrors the resolved precision")))
        (testing "a pinned INFEASIBLE schedule is rejected at compile (reject-before-emit)"
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"register budget exceeded"
               (pl/compile-gpu-program step-var :ze:0 :dtype :float :on-non-resident :nil
                                       :schedule {:stage {:space :register :copies {:a 2 :b 2}}}))))))))
