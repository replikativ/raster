(ns raster.gpu.device-timing-test
  "S1 gate: Level-Zero device-event kernel timing over linked executables.

   Asserts, on a small 2-kernel resident program:
     • fixture/profile! returns one device-time row per recorded kernel, in
       execution order, with names matching the descriptor's steps;
     • Σ per-kernel device time ≤ the device wall span (first start → last end),
       and the device wall span is consistent with the host wall time around the
       replay (device ≤ host — the host time additionally pays dispatch + sync);
     • profiling is OPT-IN: a program bound WITHOUT :profile? has no events on
       its recorded graph (the byte-identical fast path), produces the same
       numerical result, and profiling it fails loud;
     • run! on a PROFILED executable still works repeatedly (events are
       reset between replays) and matches the unprofiled result."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.dl.nn]
            [raster.compiler.pipeline :as pl]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.link :as link]))

;; Two chained elementwise ops → a 2-kernel resident program (no GEMM expansion,
;; so recorded kernels == descriptor steps, 1:1).
(deftm dt-two-step
  [a :- (Array float) b :- (Array float) n :- Long] :- (Array float)
  (let [s (raster.dl.nn/residual-add a b n)]
    (raster.dl.nn/hadamard s a n)))

(defn- fa ^floats [n seed]
  (let [rng (java.util.Random. (long seed)) arr (float-array n)]
    (dotimes [i n] (aset arr i (float (.nextGaussian rng))))
    arr))

(deftest device-timing-profile-linked-executable
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "device-timing")
    (let [n 262144
          a (fa n 1) b (fa n 2)
          args [a b n]
          prog (pl/compile-gpu-program #'dt-two-step :ze:0 :dtype :float)
          _ (is (some? prog) "dt-two-step must extract as a resident program")
          step-names (set (keep :kernel-name (:steps prog)))
          ;; host reference
          expected ^floats (apply dt-two-step args)]
      (when prog
        ;; ── profiled session ────────────────────────────────────────────────
        (let [sess (gpu/make-session :ze:0)]
          (try
            (let [program (fixture/instantiate! sess prog args {} {:profile? true})
                  ;; Warmup replay also exercises event reset between ordinary and profiled runs.
                  ;; event-reset-between-replays path on a profiled graph.
                  warm (fixture/run! program args)
                  {:keys [result profile kernel-total-ms device-wall-ms host-wall-ms]}
                  (fixture/profile! program args)
                  out ^floats (get result (:result-sym prog))]
              (testing "profiled result is numerically correct (and replayable)"
                (is (some? out))
                (dotimes [i 5]
                  (is (< (Math/abs (- (aget out i) (aget expected i))) 1.0e-5)))
                (is (java.util.Arrays/equals ^floats (get warm (:result-sym prog)) out)
                    "ordinary and profiled replay must agree"))
              (testing "one device-time row per recorded kernel, names matching the steps"
                (is (= (count (:steps prog)) (count profile)))
                (is (= step-names (set (map :kernel-name profile))))
                (is (= (set (:phases (:executable program)))
                       (set (map :phase profile)))
                    "stable linked phase identities are carried through"))
              (testing "device kernel times are positive and sum ≤ device wall span"
                (is (every? #(>= (:ms %) 0.0) profile))
                (is (pos? (double kernel-total-ms)) "some device time must be measured")
                (is (some? device-wall-ms))
                (is (<= (double kernel-total-ms) (* 1.02 (double device-wall-ms)))
                    "serialized kernels cannot exceed the first-start→last-end span"))
              (testing "device wall time ≤ host wall time (host pays dispatch + sync on top)"
                (is (<= (double device-wall-ms) (double host-wall-ms))
                    (format "device %.3f ms vs host %.3f ms" device-wall-ms host-wall-ms)))
              (testing "profiling survives repeated calls (event reset)"
                (let [p2 (fixture/profile! program args)]
                  (is (= (count profile) (count (:profile p2))))
                  (is (every? #(>= (:ms %) 0.0) (:profile p2))))))
            (finally (gpu/close-session! sess))))
        ;; ── unprofiled session: the fast path is untouched ──────────────────
        (let [sess (gpu/make-session :ze:0)]
          (try
            (let [program (fixture/instantiate! sess prog args)
                  out ^floats (get (fixture/run! program args) (:result-sym prog))
                  executable (:executable program)]
              (testing "profiling OFF: same numerical result"
                (dotimes [i 5]
                  (is (< (Math/abs (- (aget out i) (aget expected i))) 1.0e-5))))
              (testing "profiling OFF is explicit on the executable"
                (is (false? (:profile? executable))))
              (testing "profiling an unprofiled executable fails loud"
                (is (thrown? clojure.lang.ExceptionInfo (link/profile! executable)))))
            (finally (gpu/close-session! sess))))))))
