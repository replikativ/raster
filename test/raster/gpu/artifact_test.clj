(ns raster.gpu.artifact-test
  "S4 acceptance tests for the artifact-as-value layer (raster.gpu.value + raster.gpu.compiled).

   CPU-only (always run):
     A2 — donation-invalidation state machine (device value ownership discipline).
     A5 — inspection: explain / cache-key over a Compiled record (no device).

   Device-gated (Level-Zero, via the gpu_grad_parity honesty gate):
     A1 — the gemma LoRA resident train-step ported to `(r/compile …)` + `(step …)`, threaded
          as device values across 25 steps with ZERO host download in the loop, producing
          adapters BIT-IDENTICAL to the raw bind-program!/run-program! path (A6 no-regression),
          and a decreasing loss trajectory.
     A3 — frozen weights are :constant (captured at bind), never per-call inputs.
     A4 — multi-output: the out-tree projects all 14 donated adapters as DeviceArrays."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.gpu.value :as v]
            [raster.gpu.compiled :as r]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.dl.gemma-train-resident-test :as g]
            [raster.compiler.pipeline :as pl]))

;; ── access the gemma harness (privates) ──────────────────────────────────────────
(defn- gv [sym] @(ns-resolve 'raster.dl.gemma-train-resident-test sym))

;; ════════════════════════════════════════════════════════════════════════════════
;; A2 — donation-invalidation (CPU, no device)
;; ════════════════════════════════════════════════════════════════════════════════

(defrecord ^:private FakeBuf [n-elements dtype])

(deftest a2-donation-invalidation
  (testing "an ::owned value is live and readable until consumed"
    (let [inp (v/wrap-owned (->FakeBuf 8 :float) :ze:0 :float [2 4])]
      (is (v/live? inp))
      (let [buf (v/consume! inp)]
        (is (instance? FakeBuf buf) "consume! returns the buffer for the output value")
        (is (not (v/live? inp)) "input is dead after donation")
        (is (thrown? clojure.lang.ExceptionInfo (v/->host inp))
            "reading a consumed value throws use-after-free")
        (is (thrown? clojure.lang.ExceptionInfo (v/consume! inp))
            "double-consume throws")
        (let [out (v/donate-output buf :ze:0 :float [8])]
          (is (v/live? out) "output value is live")
          (is (identical? buf (:buffer out)) "output reuses the donated buffer (no copy)")))))
  (testing "::aliased free! never frees the base buffer"
    (let [base (v/wrap-owned (->FakeBuf 6 :float) :ze:0 :float [6])
          al   (v/alias-of base [2 3] :float)]
      (is (identical? (:buffer base) (:buffer al)))
      (v/free! al)
      (is (not (v/live? al)))
      (is (v/live? base) "base survives an alias free!")))
  (testing "unknown backend fails loud"
    (is (thrown? clojure.lang.ExceptionInfo (v/->device (float-array 3) :cuda:0)))))

;; ════════════════════════════════════════════════════════════════════════════════
;; A5 — inspection over a Compiled record (CPU, no device)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest a5-inspection
  (let [descriptor {:all-params '[a b n]
                    :array-params '[a b]
                    :array-roles '{a :input b :output}
                    :result-sym 'b
                    :dtype :float
                    :steps [{:convention :map} {:convention :gemm} {:convention :map}]}
        c (r/map->Compiled
           {:session nil :handle nil
            :in-tree  [{:key :a :sym 'a :role :input :donate? false :shape [4] :dtype :float}]
            :out-tree [{:key :b' :sym 'b :shape [4] :dtype :float :from :donated}]
            :donated  {:a :a'}
            :schedule nil :target :ze:0 :descriptor descriptor :args nil})]
    (testing "explain returns the artifact unchanged and prints the shape"
      (is (identical? c (r/explain c))))
    (testing "ir dumps the resident steps"
      (is (= 3 (count (r/ir c))))
      (is (= [:map :gemm :map] (mapv :convention (r/ir c)))))
    (testing "cache-key captures the serializable identity (closures excluded)"
      (let [k (r/cache-key c)]
        (is (= :ze:0 (:target k)))
        (is (= {:map 2 :gemm 1} (:steps k)))
        (is (= {:a :a'} (:donated k)))
        (is (= [4] (-> k :in-tree first :shape)))))))

;; ════════════════════════════════════════════════════════════════════════════════
;; A1 / A3 / A4 — the device acceptance: gemma resident train-step as a VALUE
;; ════════════════════════════════════════════════════════════════════════════════

(def ^:private frozen-syms
  '[x input-ln q-norm k-norm post-attn pre-ffn post-ffn Wq Wk Wv Wo Wg Wu Wd tgt])

(deftest a1-gemma-resident-train-step-as-value
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "S4 gemma resident train-step as a value")
    (let [cfg      (gv 'CFG)
          adapters (gv 'adapter-syms)
          init-state (ns-resolve 'raster.dl.gemma-train-resident-test 'init-state)
          train-args (ns-resolve 'raster.dl.gemma-train-resident-test 'train-args)
          host-loss  (ns-resolve 'raster.dl.gemma-train-resident-test 'host-loss)
          lr 0.02
          n-steps 25
          st0  (init-state cfg)
          args (train-args cfg st0 lr)
          ;; ── raw reference path: bind-program! + N× run-program! ──
          gpu  (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session   (ns-resolve gpu 'make-session)
          bind-program!  (ns-resolve gpu 'bind-program!)
          run-program!   (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          download       (ns-resolve gpu 'download)
          raw-prog (pl/compile-gpu-program #'raster.dl.gemma-train-resident-test/gblk-train-step
                                           :ze:0 :dtype :float
                                           :on-non-resident :nil :gemm-precision :f32-scalar)
          _ (is (some? raw-prog) "gblk-train-step must extract fully resident")
          raw-final
          (let [sess (make-session :ze:0)]
            (try
              (bind-program! sess raw-prog args
                             (merge (zipmap adapters (repeat :state))
                                    (zipmap frozen-syms (repeat :constant))))
              (dotimes [_ n-steps] (run-program! sess raw-prog args))
              (reduce (fn [m s] (assoc m s (download sess (keyword (name s))))) {} adapters)
              (finally (close-session! sess))))
          ;; ── artifact-as-value path: r/compile + N× (step {}) ──
          step (r/compile #'raster.dl.gemma-train-resident-test/gblk-train-step
                          args
                          {:target :ze:0 :dtype :float
                           :donate adapters :constants frozen-syms
                           :gemm-precision :f32-scalar})]
      (try
        (testing "A3: frozen weights are :constant, never per-call inputs"
          (let [roles (into {} (map (juxt :sym :role)) (:in-tree step))]
            (is (every? #(= :constant (roles %)) frozen-syms)
                "all frozen syms derive to :constant")
            (is (every? #(= :state (roles %)) adapters)
                "all adapters derive to :state (donated)")))
        (let [;; 25 resident steps — NO host download inside the loop (residency proof)
              last-out (loop [k 0 out nil]
                         (if (= k n-steps)
                           out
                           (recur (inc k) (step {}))))]
          (testing "A4: out-tree projects all 14 donated adapters (+ the result node) as DeviceArrays"
            (doseq [s adapters]
              (is (v/device-array? (get last-out (keyword (str (name s) "'"))))
                  (str s "' must be projected as a device-resident DeviceArray")))
            (is (>= (count (keys last-out)) (count adapters))
                "multi-output: at least one node per donated adapter")
            (is (every? v/device-array? (vals last-out))
                "every output is a DeviceArray (device-resident, not a host array)"))
          (testing "A1/A6: the value path is BIT-IDENTICAL to the raw run-program! path"
            (doseq [s adapters]
              (let [art (v/->host (get last-out (keyword (str (name s) "'"))))
                    raw (get raw-final s)]
                (is (java.util.Arrays/equals ^floats art ^floats raw)
                    (str s ": artifact adapters must match the raw path bit-for-bit")))))
          (testing "A1: the trained adapters differ from init (learning happened)"
            (doseq [s adapters]
              (is (not (java.util.Arrays/equals
                        ^floats (v/->host (get last-out (keyword (str (name s) "'"))))
                        ^floats (get st0 s)))
                  (str s " must change after 25 on-device steps"))))
          (testing "A1: loss decreased over the resident trajectory"
            (let [final-st (reduce (fn [m s]
                                     (assoc m s (v/->host (get last-out (keyword (str (name s) "'"))))))
                                   st0 adapters)
                  l0 (host-loss cfg st0)
                  lN (host-loss cfg final-st)]
              (println "  [S4 artifact] loss" (format "%.6f → %.6f" l0 lN))
              (is (< lN (* 0.7 l0)) (str "final " lN " vs initial " l0)))))
        (finally (r/close! step))))))
