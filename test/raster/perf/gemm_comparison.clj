(ns raster.perf.gemm-comparison
  "Opt-in public composed/explicit GEMM epilogue comparison. No schedule is promoted."
  (:refer-clojure :exclude [run!])
  (:require [raster.perf.production-canary :as canary]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.runtime.hardware :as hardware]))

(defn validate-options!
  "Bound reference work and sampling before compilation or allocation. These are probe bounds,
   not estimates of total compiler/driver memory."
  [{:keys [shape compiler-revision environment-tag rounds warmup-rounds] :as options}]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (every? #(and (string? %) (seq %)) [compiler-revision environment-tag])
                 (integer? rounds) (<= 2 rounds 120) (even? rounds)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30) (even? warmup-rounds))
    (throw (ex-info "comparison requires positive [m n k], identities and bounded even round counts"
                    {:options options})))
  (let [[m n k] shape
        work (*' m n k)
        bytes (*' 8 (+' (*' m k) (*' k n) (*' m n)))]
    (when (or (> work 8000000) (> bytes (* 64 1024 1024)))
      (throw (ex-info "comparison exceeds reference-work or logical-buffer budget"
                      {:host-products work :logical-candidate-bytes bytes}))))
  options)

(defn run!
  "Compare public dynamic GEMM→map composition with an explicit typed epilogue.
   Replay is host-synchronized, NOT pure device time. Poisoning and exact dyadic-reference
   validation run outside every timed replay. Both candidates are compiled and retained once."
  [{:keys [target shape gemm-precision compiler-revision environment-tag rounds warmup-rounds]
    :or {target :ocl:0 shape [8 256 256] gemm-precision :f32-scalar
         rounds 12 warmup-rounds 4}}]
  (validate-options! {:shape shape :compiler-revision compiler-revision
                      :environment-tag environment-tag :rounds rounds :warmup-rounds warmup-rounds})
  (hardware/init!)
  (let [arguments (canary/gemm-arguments shape)
        expected (mapv #(max (float 0.0) %)
                       (canary/gemm-reference (first arguments) (second arguments) shape))
        poison (float-array (repeat (count expected) Float/NaN))
        live (atom [])]
    (try
      (doseq [variant [:relu-composed :relu]]
        (let [start (System/nanoTime)
              prepared (canary/prepare-gemm target arguments shape
                         {:variant variant :gemm-precision gemm-precision})
              compile-ns (- (System/nanoTime) start)
              start (System/nanoTime)
              instance (compiled/instantiate! prepared)]
          ;; Register immediately so later evidence/validation failures release this instance too.
          (swap! live conj {:id variant :instance instance :prepared prepared
                           :compile-ns compile-ns :bind-ns (- (System/nanoTime) start)})))
      (when-not (apply = (map #(get-in % [:prepared :schedule :precision]) @live))
        (throw (ex-info "comparison candidates have different numerical policies" {})))
      (let [samplers
            (mapv
             (fn [{:keys [id instance]}]
               (let [resident (:executable instance)
                     output (some #(when (= 'C (:sym %)) %) (:out-tree instance))]
                 (when-not output
                   (throw (ex-info "comparison has no semantic C output" {:candidate id})))
                 {:id id
                  :sample-fn
                  (fn []
                    (link/upload! resident (:node output) poison)
                    (let [start (System/nanoTime)
                          _ (link/run! resident)
                          duration (- (System/nanoTime) start)
                          actual (vec (link/download resident (:node output)))]
                      (when-not (= expected actual)
                        (throw (ex-info "public GEMM comparison failed its poisoned-output oracle"
                                        {:candidate id :expected-head (take 8 expected)
                                         :actual-head (take 8 actual)})))
                      duration))})) @live)
            ;; Validate both candidates before any reported timing, even with zero warmup rounds.
            _ (doseq [{:keys [sample-fn]} samplers] (sample-fn))
            comparison (measurement/measure-interleaved!
                        samplers :rounds rounds :warmup-rounds warmup-rounds
                        :timing-source :host-synchronized-replay)]
        {:kind :public-gemm-epilogue-comparison :version 1
         :compiler-revision compiler-revision :environment-tag environment-tag
         :target target :device (hardware/device-signature target)
         :java-version (System/getProperty "java.vm.version")
         :shape shape :dtype :float
         :numerical-policy (get-in (first @live) [:prepared :schedule :precision])
         :input-recipe {:a :index-mod13-minus6-over8 :b :index-mod11-minus5-over8}
         :validation {:passed? true :comparison :exact :output-reset :nan
                      :oracle :host-double-dot-then-float-relu :every-replay? true}
         :scope {:public-compiler-path? true :timing-source :host-synchronized-replay
                 :cache-state :warm-resident :transfers-included? false
                 :validation-included? false :promotion? false}
         :candidates (mapv (fn [{:keys [id prepared instance compile-ns bind-ns]}]
                             {:id id :compile-ns compile-ns :bind-ns bind-ns
                              :compilation (canary/compilation-evidence prepared)
                              :execution (compiled/ir instance)}) @live)
         :comparison (update comparison :measurements
                             #(update-vals % (fn [value] (into {} value))))})
      (finally
        (doseq [{:keys [instance]} (reverse @live)] (compiled/close! instance))))))
