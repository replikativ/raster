(ns raster.perf.q4-comparison
  "Opt-in, bit-exact comparison of the generated cooperative Q4_K schedule with the serial
   source schedule. Both candidates use the public Compiled lifecycle; no result promotes a
   schedule or changes runtime policy."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.reference.ggml-q4 :as serial-reference]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.quant.ggml :as ggml]
            [raster.quant.ggml-kernels :as kernels]
            [raster.runtime.hardware :as hardware]))

(def candidate-specs
  [{:id :generated-product
    :entry #'kernels/qdot-q4-K-rows!
    :compiler :equation-first}
   {:id :serial-source
    :entry #'serial-reference/serial-qdot-q4-K-rows!
    :compiler :resident-descriptor}])

(defn validate-options!
  [{:keys [shape compiler-revision environment-tag rounds warmup-rounds] :as options}]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %) (<= % Integer/MAX_VALUE)) shape)
                 (zero? (mod (second shape) 256))
                 (every? #(and (string? %) (seq %)) [compiler-revision environment-tag])
                 (integer? rounds) (<= 2 rounds 120) (even? rounds)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30) (even? warmup-rounds))
    (throw (ex-info "Q4 comparison requires [rows in out], in divisible by 256, identities, and bounded even rounds"
                    {:options options})))
  (let [[rows in out] shape
        source-elements (+' (*' rows in) (*' out in))
        output-elements (*' rows out)]
    (when (or (> source-elements 2000000) (> output-elements 1000000))
      (throw (ex-info "Q4 comparison exceeds source or output work budget"
                      {:source-elements source-elements :output-elements output-elements}))))
  options)

(defn- deterministic-values [n multiplier scale]
  (float-array
   (map (fn [i] (float (* scale (- (mod (* multiplier i) 251) 125)))) (range n))))

(defn- row-bytes [^bytes blocks row row-size]
  (java.util.Arrays/copyOfRange blocks (* row row-size) (* (inc row) row-size)))

(defn problem
  "Create one deterministic Q4_K weights × Q8_K activations problem and its ggml bit oracle.
   Shape is [activation-rows input-width output-width]."
  [shape]
  (let [[rows in out] shape
        weights (deterministic-values (* out in) 43 0.003)
        activations (deterministic-values (* rows in) 47 0.01)
        weight-blocks (ggml/quantize :q4_K weights in out)
        activation-blocks (ggml/quantize :q8_K activations in rows)
        weight-layout (ggml/kernel-layout :q4_K weight-blocks in out)
        activation-layout (ggml/kernel-layout :q8_K activation-blocks in rows)
        weight-row-bytes (ggml/row-bytes :q4_K in)
        activation-row-bytes (ggml/row-bytes :q8_K in)
        expected-bits
        (mapv (fn [row output]
                (Float/floatToRawIntBits
                 (float (ggml/vec-dot :q4_K
                                      (row-bytes weight-blocks output weight-row-bytes)
                                      (row-bytes activation-blocks row activation-row-bytes)
                                      in))))
              (mapcat #(repeat out %) (range rows))
              (cycle (range out)))
        arguments [(:q activation-layout) (:d activation-layout) (:bsums activation-layout)
                   (:q weight-layout) (:d weight-layout) (:dmin weight-layout)
                   (:sc weight-layout) (:m weight-layout)
                   (float-array (* rows out)) (long in) (long out) (long rows)]]
    {:shape shape :arguments arguments :expected-bits expected-bits}))

(defn prepare-candidate
  [target {:keys [entry compiler]} arguments]
  (compiled/lower entry arguments
                  {:compiler compiler :target target :dtype :float :outputs '[y]
                   :constants '[xq xd xbs wq wd wdmin wsc wm]}))

(defn- output-entry [instance]
  (or (some #(when (= 'y (:sym %)) %) (:out-tree instance))
      (throw (ex-info "Q4 comparison candidate has no semantic y output" {}))))

(defn- validate-profile! [candidate expected-bits profile]
  (let [actual (get-in profile [:result :y])
        actual-bits (when actual (mapv #(Float/floatToRawIntBits %) actual))]
    (when-not (= expected-bits actual-bits)
      (throw (ex-info "Q4 comparison candidate failed the bit-exact ggml oracle"
                      {:candidate candidate :expected-head (take 8 expected-bits)
                       :actual-head (take 8 actual-bits)})))
    (let [wall-ms (:device-wall-ms profile)]
      (when-not (and (number? wall-ms) (Double/isFinite (double wall-ms))
                     (not (neg? (double wall-ms))))
        (throw (ex-info "Q4 comparison requires a finite aggregate device-event span"
                        {:candidate candidate :profile (dissoc profile :result)})))
      (* 1.0e6 (double wall-ms)))))

(defn run!
  "Compile, validate and interleave the generated and serial Q4_K schedules.

   The default Gemma-shaped case is [1 1024 640]. Inputs stay warm and resident; output poisoning,
   downloads, compilation and binding are outside the measured aggregate device-event interval."
  [{:keys [target shape compiler-revision environment-tag rounds warmup-rounds]
    :or {target :ze:0 shape [1 1024 640] rounds 12 warmup-rounds 4}}]
  (validate-options! {:shape shape :compiler-revision compiler-revision
                      :environment-tag environment-tag :rounds rounds
                      :warmup-rounds warmup-rounds})
  (hardware/init!)
  (let [{:keys [arguments expected-bits]} (problem shape)
        poison (float-array (repeat (count expected-bits) Float/NaN))
        live (atom [])
        profiles (atom [])]
    (try
      (doseq [{:keys [id compiler] :as candidate} candidate-specs]
        (let [started (System/nanoTime)
              prepared (prepare-candidate target candidate arguments)
              compile-ns (- (System/nanoTime) started)
              started (System/nanoTime)
              instance (compiled/instantiate! prepared {:profile? true})]
          (swap! live conj {:id id :compiler compiler :prepared prepared :instance instance
                           :compile-ns compile-ns :bind-ns (- (System/nanoTime) started)})))
      (let [samplers
            (mapv (fn [{:keys [id instance]}]
                    (let [resident (:executable instance)
                          output (output-entry instance)
                          invocation (atom -1)]
                      {:id id
                       :sample-fn
                       (fn []
                         (let [replay-index (swap! invocation inc)
                               sampling-phase (cond (zero? replay-index) :validation
                                                    (<= replay-index warmup-rounds) :warmup
                                                    :else :measurement)
                               sample-index (when (= :measurement sampling-phase)
                                              (- replay-index warmup-rounds 1))]
                           (link/upload! resident (:node output) poison)
                           (let [profile (compiled/profile instance)
                                 duration (validate-profile! id expected-bits profile)]
                             (swap! profiles conj
                                    {:candidate id :replay-index replay-index
                                     :sampling-phase sampling-phase :sample-index sample-index
                                     :profile (dissoc profile :result)})
                             duration)))}))
                  @live)
            _ (doseq [{:keys [sample-fn]} samplers] (sample-fn))
            comparison (measurement/measure-interleaved!
                        samplers :rounds rounds :warmup-rounds warmup-rounds
                        :timing-source :device-event)]
        {:kind :public-q4-k-schedule-comparison :version 1
         :compiler-revision compiler-revision :environment-tag environment-tag
         :target target :device (hardware/device-signature target)
         :java-version (System/getProperty "java.vm.version")
         :shape shape :weight-format :q4_K :activation-format :q8_K :dtype :float
         :validation {:passed? true :comparison :raw-float-bits
                      :oracle :raster.quant.ggml/vec-dot :every-replay? true
                      :output-reset :nan}
         :scope {:public-compiled-lifecycle? true :timing-source :device-event
                 :cache-state :warm-resident :transfers-included? false
                 :validation-included? false :promotion? false}
         :replay-profiles @profiles
         :candidates
         (mapv (fn [{:keys [id compiler prepared instance compile-ns bind-ns]}]
                 {:id id :compiler compiler :compile-ns compile-ns :bind-ns bind-ns
                  :execution (compiled/ir instance)
                  :resident-step-count (count (get-in prepared [:descriptor :steps]))})
               @live)
         :comparison (update comparison :measurements #(update-vals % (fn [m] (into {} m))))})
      (finally
        (doseq [{:keys [instance]} (reverse @live)] (compiled/close! instance))))))
