(ns raster.gpu.gate-honesty-test
  "Regression gate for the GPU test-gate HONESTY property (CPU-only — needs no GPU).

   The GPU-gated tests used to probe availability with `(catch Throwable _ false)`, so
   a BROKEN ze-runtime load looked identical to 'no GPU device': every gated deftest
   took its skip branch and the whole GPU suite went green with zero assertions — a
   suite that reported success having tested nothing. `raster.dl.gpu-grad-parity`'s
   honest probe + `gpu-skip!` fix that. These tests pin the contract by driving
   `gpu-skip!` under each simulated `gpu-status` and asserting what it reports:

     :load-failed  → a FAILING assertion (a runtime breakage surfaces, never a silent skip)
     :no-device    → exactly one PASSING marker (visible, deterministic count)
     :probe-error  → one PASSING marker + a warning (won't redden a GPU-less box)

   This is also the standing proof that 'simulate a broken GPU load' surfaces loudly."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gpu-grad-parity :as gp]))

(defn- capture-skip
  "Run (gp/gpu-skip! label) with gpu-status pinned to `status-map`, capturing the
   clojure.test report event types it emits WITHOUT touching the outer suite counters.
   Returns a vector of :pass/:fail/:error keywords."
  [status-map]
  (let [events (atom [])]
    (with-redefs [gp/gpu-status (delay status-map)
                  clojure.test/report (fn [m] (swap! events conj (:type m)))]
      (gp/gpu-skip! "gate-honesty-probe"))
    @events))

(deftest load-failure-fails-loud
  (testing "a THROWN ze-runtime load registers a FAILING assertion, not a silent skip"
    (let [events (capture-skip {:status :load-failed
                                :error (Exception. "simulated ze-runtime load failure")})]
      (is (some #{:fail} events)
          (str "gpu-skip! on :load-failed must emit a :fail — got " (pr-str events)))
      (is (not-any? #{:pass} events)
          "a broken load must NOT register a passing marker (that would hide the breakage)"))))

(deftest no-device-registers-visible-marker
  (testing "genuinely absent GPU registers exactly one PASSING marker (stable count)"
    (let [events (capture-skip {:status :no-device})]
      (is (= [:pass] events)
          (str "gpu-skip! on :no-device must emit exactly one :pass marker — got "
               (pr-str events))))))

(deftest probe-error-warns-but-does-not-fail
  (testing "query-devices throwing (no L0 loader) warns + marks, but does not fail CI"
    (let [events (capture-skip {:status :probe-error
                                :error (Exception. "no Level-Zero loader")})]
      (is (= [:pass] events)
          (str "gpu-skip! on :probe-error must emit exactly one :pass marker — got "
               (pr-str events)))
      (is (not-any? #{:fail} events) ":probe-error must not redden a GPU-less machine"))))
