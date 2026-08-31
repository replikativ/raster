(ns raster.gpu.artifact-test
  "S4 acceptance tests for the artifact-as-value layer (raster.gpu.value + raster.gpu.compiled).

   CPU-only (always run):
     A2 — donation-invalidation state machine (device value ownership discipline).
     A5 — inspection: explain / cache-key over a Compiled record (no device).

   Device-gated (Level-Zero, via the gpu_grad_parity honesty gate):
     A1 — the gemma LoRA resident train-step ported to `(r/compile …)` + `(step …)`, threaded
          as device values across 25 steps with ZERO host download in the loop, producing
          adapters BIT-IDENTICAL across independent certified executions (A6 determinism),
          and a decreasing loss trajectory.
     A3 — frozen weights are :constant (captured at bind), never per-call inputs.
     A4 — multi-output: the out-tree projects all 14 donated adapters as DeviceArrays."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.dl.gemma-train-resident-test :as g]
            [raster.dl.nn :as nn]
            [raster.gpu.compiled :as r]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.value :as v]))

;; ── access the gemma harness (privates) ──────────────────────────────────────────
(defn- gv [sym] @(ns-resolve 'raster.dl.gemma-train-resident-test sym))

;; ════════════════════════════════════════════════════════════════════════════════
;; A2 — donation-invalidation (CPU, no device)
;; ════════════════════════════════════════════════════════════════════════════════

(defrecord ^:private FakeBuf [n-elements dtype])

(deftm artifact-two-step
  [x :- (Array float) bias :- (Array float) n :- Long] :- (Array float)
  (let [sum (nn/residual-add x bias n)]
    (nn/hadamard sum x n)))

(deftest a2-donation-invalidation
  (testing "an ::owned value is live and readable until consumed"
    (let [inp (v/wrap-owned (->FakeBuf 8 :float) :ze:0 :float [2 4])]
      (is (v/live? inp))
      (let [donation (v/consume! inp)]
        (is (v/donated-buffer? donation) "consume! returns an opaque donation token")
        (is (instance? FakeBuf (:buffer donation)) "the token retains the resident buffer")
        (is (not (v/live? inp)) "input is dead after donation")
        (is (thrown? clojure.lang.ExceptionInfo (v/->host inp))
            "reading a consumed value throws use-after-free")
        (is (thrown? clojure.lang.ExceptionInfo (v/consume! inp))
            "double-consume throws")
        (let [out (v/donate-output donation :float [8])]
          (is (v/live? out) "output value is live")
          (is (identical? (:buffer donation) (:buffer out)) "output reuses the buffer (no copy)")
          (is (= (get-in donation [:view :allocation :id])
                 (get-in out [:view :allocation :id]))
              "donation preserves allocation identity")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already claimed"
                                (v/donate-output donation :float [8]))
              "one token cannot create two owners")))))
  (testing "::aliased free! never frees the base buffer"
    (let [base (v/wrap-owned (->FakeBuf 6 :float) :ze:0 :float [6])
          al   (v/alias-of base {:shape [2 3] :dtype :float})]
      (is (identical? (:buffer base) (:buffer al)))
      (is (bview/same-range? (:view base) (:view al)))
      (v/free! al)
      (is (not (v/live? al)))
      (is (v/live? base) "base survives an alias free!")))
  (testing "an alias can name a checked byte range of its base"
    (let [base (v/wrap-owned (->FakeBuf 8 :float) :ze:0 :float [6])
          slice (v/alias-of base {:byte-offset 8 :shape [3]})]
      (is (= 8 (get-in slice [:view :byte-offset])))
      (is (= [3] (:shape slice)))
      (is (= (get-in base [:view :allocation :id])
             (get-in slice [:view :allocation :id])))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its base"
                            (v/alias-of base {:byte-offset 20 :shape [2]})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot transfer"
                            (v/consume! slice)))))
  (testing "a failed ->device shape contract releases the allocation it just made"
    (let [buffer (->FakeBuf 2 :float)
          freed (atom [])
          resolver (fn [_ name]
                     (case name
                       "buffer-of-array" (fn [_] buffer)
                       "free-buffer!" #(swap! freed conj %)
                       (throw (ex-info "unexpected runtime call" {:name name}))))]
      (with-redefs-fn
        {(ns-resolve 'raster.gpu.value 'rt-fn) resolver}
        (fn []
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its allocation"
                                (v/->device (float-array 2) :ze:0 {:shape [3]})))
          (is (= [buffer] @freed))))))
  (testing "unknown backend fails loud"
    (is (thrown? clojure.lang.ExceptionInfo (v/->device (float-array 3) :cuda:0)))))

(deftest a2-device-alias-downloads-only-its-view
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "DeviceArray ranged alias")
    (let [base (v/->device (float-array [0.0 1.0 2.0 3.0 4.0 5.0]) :ze:0)
          slice (v/alias-of base {:byte-offset 8 :shape [3]})]
      (try
        (is (= [2.0 3.0 4.0] (vec (v/->host slice))))
        (finally
          (v/free! slice)
          (v/free! base))))))

(deftest compiled-device-input-never-round-trips-through-host
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "Compiled LinkedExecutable device input")
    (let [n 64
          bias (float-array (repeat n 2.0))
          initial (float-array n)
          input (v/->device (float-array (repeat n 3.0)) :ze:0)
          compiled (r/compile #'artifact-two-step [initial bias n]
                              {:target :ze:0 :constants '[bias]
                               :on-non-resident :throw :profile? true})]
      (try
        (let [result (with-redefs [v/->host
                                   (fn [_]
                                     (throw (AssertionError.
                                             "Compiled must not download a device input")))]
                       (compiled {:x input}))
              output (get result (keyword (name (:result-sym (:descriptor compiled)))))]
          (is (every? #(= 15.0 (double %)) (v/->host output)))
          (let [profile (r/profile compiled)]
            (is (= 1 (count (:profile profile))))
            (is (pos? (:device-wall-ms profile)))
            (is (not (v/live? output))
                "profiling invalidates an earlier output whose resident storage it overwrites")))
        (finally
          (v/free! input)
          (r/close! compiled))))))

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
           {:lowering nil :executable nil
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
          ;; Independent certified execution: guards deterministic lowering/runtime behavior.
          reference-step
          (r/compile #'raster.dl.gemma-train-resident-test/gblk-train-step
                     args
                     {:target :ze:0 :dtype :float
                      :donate adapters :constants frozen-syms
                      :gemm-precision :f32-scalar})
          reference-final
          (try
            (let [last-out (loop [iteration 0 output nil]
                             (if (= iteration n-steps)
                               output
                               (recur (inc iteration) (reference-step {}))))]
              (reduce (fn [values symbol]
                        (assoc values symbol
                               (v/->host (get last-out
                                              (keyword (str (name symbol) "'"))))))
                      {} adapters))
            (finally (r/close! reference-step)))
          ;; ── artifact-as-value path: r/compile + N× (step {}) ──
          step (r/compile #'raster.dl.gemma-train-resident-test/gblk-train-step
                          args
                          {:target :ze:0 :dtype :float
                           :donate adapters :constants frozen-syms
                           :gemm-precision :f32-scalar})]
      (try
        (testing "Compiled is the certified LinkPlan API, not a parallel resident binder"
          (is (resident-plan/certified-plan? (:lowering step)))
          (is (gpu-link/linked-executable? (:executable step)))
          (is (= (r/plan step) (:plan (:executable step))))
          (is (= :link-plan (get-in (r/certificate step) [:target-dialect]))))
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
                "every output is a DeviceArray (device-resident, not a host array)")
            (doseq [{:keys [key sym]} (:out-tree step)
                    :let [da (get last-out key)]
                    :when da]
              (let [node (get-in (r/certificate step) [:bindings sym])
                    resident (gpu-link/node-view (:executable step) node)
                    session (:session (:executable step))]
                (is (= (get-in @session [:allocations (:key resident) :id])
                       (get-in da [:view :allocation :id]))
                    (str key " shares the certified LinkPlan allocation identity"))
                (is (= node (get-in da [:view :allocation :id]))
                    (str key " preserves the LinkPlan allocation identity at runtime")))))
          (testing "A1/A6: independent certified executions are bit-identical"
            (doseq [s adapters]
              (let [art (v/->host (get last-out (keyword (str (name s) "'"))))
                    reference (get reference-final s)]
                (is (java.util.Arrays/equals ^floats art ^floats reference)
                    (str s ": certified executions must match bit-for-bit")))))
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
              (is (< lN (* 0.7 l0)) (str "final " lN " vs initial " l0))))
          (testing "a projected value threads through its exact donated view"
            (let [in-key (keyword (name (first adapters)))
                  out-key (keyword (str (name (first adapters)) "'"))
                  prior (step {})
                  input (get prior out-key)
                  allocation-id (get-in input [:view :allocation :id])
                  next (step {in-key input})]
              (is (not (v/live? input)) "the donated input view is consumed")
              (is (= allocation-id (get-in next [out-key :view :allocation :id]))
                  "the fresh output preserves the session allocation identity")))
          (testing "S4 boundary contract (M4): a retained prior output is invalidated by the next call"
            (let [prev    (step {})
                  prev-da (get prev (keyword (str (name (first adapters)) "'")))]
              (is (v/live? prev-da) "a freshly projected output is live")
              (step {})
              (is (not (v/live? prev-da)) "the next invocation invalidates the prior output (no silent mutation)")
              (is (thrown? clojure.lang.ExceptionInfo (v/->host prev-da))
                  "->host on a stale output fails loud")))
          (testing "S4 boundary contract (B1): unsupported / foreign inputs fail loud, never silently ignored"
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported input key"
                                  (step {:not-a-param (float-array 1)}))
                "a non-param key throws instead of being dropped")
            (let [foreign (v/->device (float-array (java.lang.reflect.Array/getLength ^floats (st0 (first adapters))))
                                      :ze:0)]
              (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not this artifact's resident view"
                                    (step {(keyword (name (first adapters))) foreign}))
                  "a foreign device value donated to a slot throws instead of being consumed-and-ignored")
              (v/free! foreign))))
        (finally (r/close! step))))))
