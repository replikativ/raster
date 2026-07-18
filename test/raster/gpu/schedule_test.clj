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
  (testing "grf256 DOUBLES the budget and GENUINELY unblocks register staging (acc is fixed, not budget-coupled)"
    ;; the accumulator is a fixed 256 B/lane tile property; grf256 lifts the budget to 512, so
    ;; register double-buffering both operands (256 acc + 256 staged = 512) now FITS — the knob is
    ;; not pointless (the earlier budget-coupled model made grf256 useless for f16-xmx).
    (let [g128 (sched/resolve (sched/derive-default nil arc-desc)
                              {:stage {:space :register :copies {:a 2 :b 2}}})
          g256 (sched/resolve (sched/derive-default nil arc-desc)
                              {:grf {:mode :grf256} :stage {:space :register :copies {:a 2 :b 2}}})]
      (is (thrown? clojure.lang.ExceptionInfo (sched/feasible? g128 arc-desc)) "rejected at grf128")
      (is (true? (sched/feasible? g256 arc-desc)) "same schedule FITS at grf256 (512 ≤ 512)")))
  (testing "deeper staging costs proportionally more (:depth is honored, not ignored)"
    (let [d1 (sched/resolve (sched/derive-default nil arc-desc)
                            {:grf {:mode :grf256} :stage {:space :register :copies {:a 2 :b 0} :depth 1}})
          d3 (sched/resolve (sched/derive-default nil arc-desc)
                            {:grf {:mode :grf256} :stage {:space :register :copies {:a 2 :b 0} :depth 3}})]
      (is (true? (sched/feasible? d1 arc-desc)) "depth 1: 256 + 2×1×64 = 384 ≤ 512")
      (is (thrown? clojure.lang.ExceptionInfo (sched/feasible? d3 arc-desc))
          "depth 3: 256 + 2×3×64 = 640 > 512")))
  (testing "the DERIVED tile is GRF-bound (always fits); a user-PINNED oversized tile is rejected"
    ;; T3: derive-gemm-tile GRF-bounds the accumulator, so a smaller-budget device gets a SMALLER
    ;; tile that fits by construction — the default can no longer spill. The gate's live job is to
    ;; reject an EXPLICIT tile a caller pinned that exceeds this device's budget.
    (let [small (assoc arc-desc :grf-bytes-per-lane 128)]
      (is (true? (sched/feasible? (sched/derive-default nil small) small))
          "the derived default tile fits the 128 B/lane device (GRF-bound, no spill)")
      (let [pinned (sched/resolve (sched/derive-default nil small)
                                  {:tile {:block-m 128 :block-n 128 :sg-m 32 :sg-n 32 :block-k 32
                                          :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}})]
        (is (thrown? clojure.lang.ExceptionInfo (sched/feasible? pinned small))
            "a pinned 32×32 tile (256 B/lane acc) does NOT fit a 128 B/lane budget"))))
  (testing "unmodeled precision / stage-space FAIL LOUD (not a silent XMX bind)"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unknown :precision"
                          (sched/feasible? (sched/resolve (sched/derive-default nil arc-desc)
                                                          {:precision :bf16}) arc-desc)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unknown :stage :space"
                          (sched/feasible? (sched/resolve (sched/derive-default nil arc-desc)
                                                          {:stage {:space :regsiter}}) arc-desc))))
  (testing "conflicting :gemm-precision sugar and :precision throw, not silently prefer the deprecated key"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"conflicting"
                          (sched/resolve (sched/derive-default nil arc-desc)
                                         {:gemm-precision :f16-xmx :precision :f32-scalar})))))

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
