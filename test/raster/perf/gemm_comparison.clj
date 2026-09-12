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
  [{:keys [shape compiler-revision environment-tag rounds warmup-rounds timing-source composed-variant
           input-policy]
    :as options}]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (every? #(and (string? %) (seq %)) [compiler-revision environment-tag])
                 (contains? #{:host-synchronized-replay :device-event} timing-source)
                 (contains? #{:relu-composed :relu-prebound} composed-variant)
                 (contains? #{:constant-operands :changing-activation} input-policy)
                 (integer? rounds) (<= 2 rounds 120) (even? rounds)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30) (even? warmup-rounds))
    (throw (ex-info "comparison requires positive [m n k], identities and bounded even round counts"
                    {:options options})))
  (let [[m n k] shape
        work (*' (if (= :changing-activation input-policy) 2 1) m n k)
        bytes (*' 8 (+' (*' m k) (*' k n) (*' m n)))]
    (when (or (> work 8000000) (> bytes (* 64 1024 1024)))
      (throw (ex-info "comparison exceeds reference-work or logical-buffer budget"
                      {:host-products work :logical-candidate-bytes bytes}))))
  options)

(defn run!
  "Compare public dynamic GEMM→map composition with an explicit typed epilogue.
   :composed-variant selects :relu-composed (extent after contraction, default) or :relu-prebound
   (extent before contraction). :input-policy :changing-activation alternates A and -A every replay,
   with only B constant. Defaults to host-synchronized replay; :timing-source :device-event uses the recorded graph
   event span, never a host-time fallback. Poisoning and exact dyadic-reference
   validation run outside every timed replay. Both candidates are compiled and retained once."
  [{:keys [target shape gemm-precision compiler-revision environment-tag rounds warmup-rounds timing-source
           composed-variant input-policy]
    :or {target :ocl:0 shape [8 256 256] gemm-precision :f32-scalar
         rounds 12 warmup-rounds 4 timing-source :host-synchronized-replay
         composed-variant :relu-composed input-policy :constant-operands}}]
  (validate-options! {:shape shape :compiler-revision compiler-revision
                      :environment-tag environment-tag :rounds rounds :warmup-rounds warmup-rounds
                      :timing-source timing-source :composed-variant composed-variant
                      :input-policy input-policy})
  (hardware/init!)
  (let [arguments (canary/gemm-arguments shape)
        changing? (= :changing-activation input-policy)
        constants (if changing? ['B] ['A 'B])
        activations (cond-> [(first arguments)]
                      changing? (conj (float-array (map #(float (- %)) (first arguments)))))
        references (mapv (fn [a]
                           (mapv #(max (float 0.0) %)
                                 (canary/gemm-reference a (second arguments) shape))) activations)
        poison (float-array (repeat (count (first references)) Float/NaN))
        live (atom [])
        profiles (atom [])]
    (try
      (doseq [variant [composed-variant :relu]]
        (let [start (System/nanoTime)
              prepared (canary/prepare-gemm target arguments shape
                         {:variant variant :gemm-precision gemm-precision :constants constants})
              compile-ns (- (System/nanoTime) start)
              start (System/nanoTime)
              instance (compiled/instantiate! prepared {:profile? (= :device-event timing-source)})]
          ;; Register immediately so later evidence/validation failures release this instance too.
          (swap! live conj {:id variant :instance instance :prepared prepared
                           :compile-ns compile-ns :bind-ns (- (System/nanoTime) start)})))
      (when-not (apply = (map #(get-in % [:prepared :schedule :precision]) @live))
        (throw (ex-info "comparison candidates have different numerical policies" {})))
      (let [samplers
            (mapv
             (fn [{:keys [id instance]}]
               (let [resident (:executable instance)
                     input (some #(when (= 'A (:sym %)) %) (:in-tree instance))
                     invocation (atom -1)
                     output (some #(when (= 'C (:sym %)) %) (:out-tree instance))]
                 (when-not output
                   (throw (ex-info "comparison has no semantic C output" {:candidate id})))
                 (when (and changing? (not= :input (:role input)))
                   (throw (ex-info "changing activation requires a semantic A input"
                                   {:candidate id :input input})))
                 {:id id
                  :sample-fn
                  (fn []
                    (let [input-index (mod (swap! invocation inc) (count activations))
                          expected (nth references input-index)]
                      (when changing? (link/upload! resident (:node input) (nth activations input-index)))
                      (link/upload! resident (:node output) poison)
                      (let [duration
                            (if (= :device-event timing-source)
                              (let [profile (link/profile! resident)
                                    wall-ms (:device-wall-ms profile)]
                                (when-not (and (number? wall-ms)
                                               (Double/isFinite (double wall-ms))
                                               (not (neg? (double wall-ms))))
                                  (throw (ex-info "comparison requires a finite device event span"
                                                  {:candidate id :profile profile})))
                                (swap! profiles conj {:candidate id :input-index input-index :profile profile})
                                (* 1.0e6 (double wall-ms)))
                              (let [start (System/nanoTime)]
                                (link/run! resident)
                                (- (System/nanoTime) start)))
                            actual (vec (link/download resident (:node output)))]
                        (when-not (= expected actual)
                          (throw (ex-info "public GEMM comparison failed its poisoned-output oracle"
                                          {:candidate id :expected-head (take 8 expected)
                                           :actual-head (take 8 actual)})))
                        duration)))})) @live)
            ;; Validate both candidates before any reported timing, even with zero warmup rounds.
            _ (doseq [{:keys [sample-fn]} samplers] (sample-fn))
            comparison (measurement/measure-interleaved!
                        samplers :rounds rounds :warmup-rounds warmup-rounds
                        :timing-source timing-source)]
        {:kind :public-gemm-epilogue-comparison :version 1
         :compiler-revision compiler-revision :environment-tag environment-tag
         :target target :device (hardware/device-signature target)
         :java-version (System/getProperty "java.vm.version")
         :shape shape :dtype :float
         :numerical-policy (get-in (first @live) [:prepared :schedule :precision])
         :input-recipe {:a :index-mod13-minus6-over8 :b :index-mod11-minus5-over8
                        :activation-pattern (if changing? :alternating-sign :fixed)}
         :validation {:passed? true :comparison :exact :output-reset :nan
                      :oracle :host-double-dot-then-float-relu :every-replay? true}
         :scope {:public-compiler-path? true :timing-source timing-source
                 :input-policy input-policy :constant-operands constants
                 :cache-state :warm-resident :transfers-included? false
                 :one-time-prologue-included? false
                 :validation-included? false :promotion? false}
         ;; Includes prevalidation and warmup in actual execution order, not just timed rounds.
         :replay-profiles @profiles
         :candidates (mapv (fn [{:keys [id prepared instance compile-ns bind-ns]}]
                             {:id id :compile-ns compile-ns :bind-ns bind-ns
                              :compilation (canary/compilation-evidence prepared)
                              :execution (compiled/ir instance)}) @live)
         :comparison (update comparison :measurements
                             #(update-vals % (fn [value] (into {} value))))})
      (finally
        (doseq [{:keys [instance]} (reverse @live)] (compiled/close! instance))))))
