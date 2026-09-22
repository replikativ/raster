(ns raster.gpu.dispatch-tuning
  "Explicit offline autotuning for a KernelDispatch.

   Compilation and runtime selection never benchmark. This namespace measures already-emitted,
   ABI-compatible executable alternatives through a caller-supplied validated benchmark, converts
   winners at sampled runtime scalar values into a piecewise selector, and atomically caches it.
   Applying a result rechecks the full identity: device, emitted sources, ABI, numerical mode, and
  layout. A stale or cross-device result therefore cannot silently select a kernel."
  (:require [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.charset StandardCharsets]
           [java.security MessageDigest]))

(def tuning-version 5)

(defrecord DispatchTuning
           [key identity selector measurements])

(defn dispatch-tuning?
  "Recognize tuning values across Typed Clojure child classloaders."
  [x]
  (and x (= "raster.gpu.dispatch_tuning.DispatchTuning" (.getName (class x)))))

(defn- sha256
  [value]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256")
                        (.getBytes (str value) StandardCharsets/UTF_8))]
    (apply str (map #(format "%02x" (bit-and 0xff (int %))) digest))))

(defn- canonical-data
  [value]
  (cond
    (map? value) (into (sorted-map-by #(compare (pr-str %1) (pr-str %2)))
                       (map (fn [[key item]] [(canonical-data key) (canonical-data item)]))
                       value)
    (set? value) (mapv canonical-data (sort-by pr-str value))
    (vector? value) (mapv canonical-data value)
    (sequential? value) (mapv canonical-data value)
    :else value))

(defn executable-signature
  "Stable correctness/performance identity of one emitted executable alternative.

   For a graph, the source hash covers node order, dependencies, uses, launch contracts, private
   buffers, and every emitted module. A schedule change therefore cannot reuse stale tuning data
   merely because its public ABI stayed constant."
  [executable]
  (let [executable (kexec/validate! executable)
        graph? (= :kernel-graph (kexec/kind executable))
        schedule (if graph?
                   {:buffers {:inputs (:inputs executable)
                              :outputs (:outputs executable)
                              :temporaries (:temporaries executable)}
                    :nodes (mapv (fn [node]
                                   {:id (:id node)
                                    :uses (:uses node)
                                    :dependencies (:dependencies node)
                                    :artifact (kart/compilation-identity (select-keys (:operation node)
                                                           [:kernel-name :target :source :abi
                                                            :arguments :launch :preconditions :temporaries])
                                                                         (:operation node))})
                                 (:nodes executable))}
                   (kart/compilation-identity
                    (select-keys executable
                                 [:kernel-name :target :source :launch :preconditions :temporaries])
                    executable))]
    {:kind (kexec/kind executable)
     :strategy (kdispatch/alternative-strategy executable)
     :target (kexec/target executable)
     :entry-points (kexec/entry-points executable)
     :source-hash (sha256 (pr-str (canonical-data schedule)))
     :abi-hash (sha256 (pr-str (canonical-data (kexec/abi executable))))
     :arguments-hash (sha256 (pr-str (kexec/arguments executable)))
     :effects-hash (sha256 (pr-str (canonical-data (kexec/effects executable))))}))

(defn tuning-identity
  "Build the complete identity that guards a dispatch tuning result.

   `numerical-mode` and `layout` are required caller data because neither can be reconstructed
   safely from a generic kernel name. The sampled runtime values and improvement threshold are
   policy inputs: changing either must miss the cache. Examples are
   {:input :f16 :accumulate :f32 :output :f32} and
   {:edge-list :packed :value :row-major}."
  [dispatch descriptor runtime-values numerical-mode layout improvement-threshold]
  (let [dispatch (kdispatch/validate! dispatch)]
    (when (nil? numerical-mode)
      (throw (ex-info "dispatch tuning identity requires :numerical-mode" {})))
    (when (nil? layout)
      (throw (ex-info "dispatch tuning identity requires :layout" {})))
    (canonical-data
     {:version tuning-version
      :dispatch-id (:id dispatch)
      :selector-argument (get-in dispatch [:selector :argument])
      :device (hardware/evidence-signature descriptor)
      :numerical-mode numerical-mode
      :layout layout
      :policy {:runtime-values (vec runtime-values)
               :improvement-threshold (double improvement-threshold)}
      :alternatives (mapv executable-signature (:alternatives dispatch))})))

(defn cache-key
  [identity]
  (sha256 (pr-str (canonical-data identity))))

(declare validate-tuning-evidence!)

(defn- validated-tuning-from-data
  [dispatch identity key data]
  (when (and (= tuning-version (:version data))
             (= key (:key data))
             (= identity (:identity data))
             (map? (:selector data))
             (vector? (:measurements data)))
    (kdispatch/with-selector dispatch (:selector data))
    (validate-tuning-evidence!
     dispatch (->DispatchTuning key identity (:selector data) (:measurements data)))))

(defn tuning-data
  "Project a DispatchTuning to plain, versioned EDN data suitable for an artifact or wire envelope.

   The result deliberately retains the unhashed complete identity as well as its key. A consumer
   must call `restore-tuning`; a selector by itself is not portable tuning evidence."
  [tuning]
  (when-not (dispatch-tuning? tuning)
    (throw (ex-info "tuning-data requires a DispatchTuning" {:tuning tuning})))
  {:version tuning-version
   :key (:key tuning)
   :identity (:identity tuning)
   :selector (:selector tuning)
   :measurements (:measurements tuning)})

(defn restore-tuning
  "Restore transported plain tuning data after rederiving its complete current identity.

   Unlike a cache lookup, malformed, stale, cross-build or cross-device evidence fails loudly.
   This boundary is intended for program receipts and externally stored artifacts where silently
   treating invalid supplied evidence as a cache miss would hide a deployment error."
  [dispatch data descriptor numerical-mode layout]
  (when-not (map? data)
    (throw (ex-info "transported dispatch tuning must be a map"
                    {:reason :invalid-transported-dispatch-tuning :data data})))
  (let [{:keys [runtime-values improvement-threshold]} (get-in data [:identity :policy])]
    (when-not (and (vector? runtime-values) (number? improvement-threshold))
      (throw (ex-info "transported dispatch tuning has an invalid policy identity"
                      {:reason :invalid-transported-dispatch-tuning-policy
                       :policy (get-in data [:identity :policy])})))
    (let [identity (tuning-identity dispatch descriptor runtime-values numerical-mode layout
                                    improvement-threshold)
          key (cache-key identity)]
      (or (validated-tuning-from-data dispatch identity key data)
          (throw (ex-info "transported dispatch tuning differs from the target dispatch"
                          {:reason :transported-dispatch-tuning-identity
                           :expected-key key
                           :actual-key (:key data)
                           :expected identity
                           :actual (:identity data)}))))))

(defn cache-get
  "Read and validate a cached result for the exact dispatch identity. Corrupt/stale entries miss."
  [dispatch identity]
  (let [key (cache-key identity)]
    (try
      (validated-tuning-from-data dispatch identity key (cache/read-entry key))
      (catch Exception _ nil))))

(defn cache-put!
  "Atomically persist a DispatchTuning as plain EDN."
  [tuning]
  (when-not (dispatch-tuning? tuning)
    (throw (ex-info "dispatch tuning cache requires a DispatchTuning" {:value tuning})))
  (let [data (tuning-data tuning)]
    (cache/write-entry! (:key tuning) data)
    tuning))

(def ^:private measurement-fields
  [:min-ns :median-ns :p75-ns :mean-ns :cv :stationary? :n :warmup-iterations
   :budget-ms :cold-warm :timing-source :compile-ms :hashes])

(defn- validate-measured-result!
  [executable runtime-value result]
  (let [{:keys [measurement validation]} result
        signature (executable-signature executable)]
    (when-not (measurement/measurement? measurement)
      (throw (ex-info "dispatch benchmark must return a Measurement"
                      {:runtime-value runtime-value
                       :strategy (:strategy signature)
                       :measurement measurement})))
    (when-not (= :device-event (:timing-source measurement))
      (throw (ex-info "dispatch tuning accepts device-event measurements only"
                      {:runtime-value runtime-value
                       :strategy (:strategy signature)
                       :timing-source (:timing-source measurement)})))
    (when-not (:stationary? measurement)
      (throw (ex-info "dispatch tuning refuses a non-stationary measurement"
                      {:runtime-value runtime-value
                       :strategy (:strategy signature)
                       :cv (:cv measurement)})))
    (when-not (and (map? validation)
                   (true? (:passed? validation))
                   (string? (:oracle-hash validation))
                   (not-empty (:oracle-hash validation))
                   (= (:source-hash signature) (:candidate-hash validation)))
      (throw (ex-info "dispatch candidate must pass an oracle validation tied to its source hash"
                      {:runtime-value runtime-value
                       :strategy (:strategy signature)
                       :expected-candidate-hash (:source-hash signature)
                       :validation validation})))
    {:runtime-value runtime-value
     :strategy (:strategy signature)
     :measurement (select-keys measurement measurement-fields)
     :validation (select-keys validation
                              [:passed? :oracle-hash :candidate-hash :max-error :rtol :atol])}))

(defn- validate-benchmark-result!
  [executable runtime-value result]
  (if (= :inapplicable (:status result))
    (let [signature (executable-signature executable)
          {:keys [violations candidate-hash]} result]
      (when-not (and (= candidate-hash (:source-hash signature))
                     (vector? violations) (seq violations)
                     (every? #(and (map? %) (keyword? (:reason %))) violations)
                     (not (contains? result :measurement))
                     (not (contains? result :validation)))
        (throw (ex-info "inapplicable dispatch candidate requires source-bound preflight violations"
                        {:reason :dispatch-tuning-inapplicable-result
                         :strategy (:strategy signature) :runtime-value runtime-value
                         :result result})))
      {:runtime-value runtime-value :strategy (:strategy signature)
       :status :inapplicable :candidate-hash candidate-hash :violations violations})
    (do
      (when (contains? result :status)
        (throw (ex-info "unknown dispatch benchmark result status"
                        {:reason :dispatch-tuning-result-status :status (:status result)})))
      (validate-measured-result! executable runtime-value result))))

(defn- winner-at
  [rows default-strategy improvement-threshold]
  (let [by-strategy (into {} (map (juxt :strategy identity)) rows)
        default-row (or (get by-strategy default-strategy)
                        (throw (ex-info "dispatch measurements omit the default strategy"
                                        {:default-strategy default-strategy
                                         :strategies (set (keys by-strategy))})))
        _ (when (= :inapplicable (:status default-row))
            (throw (ex-info "dispatch tuning requires an applicable default strategy"
                            {:reason :dispatch-tuning-default-inapplicable
                             :default-strategy default-strategy
                             :runtime-value (:runtime-value default-row)
                             :violations (:violations default-row)})))
        applicable (filterv #(not= :inapplicable (:status %)) rows)
        best-row (apply min-key #(get-in % [:measurement :min-ns]) applicable)
        default-cost (double (get-in default-row [:measurement :min-ns]))
        best-cost (double (get-in best-row [:measurement :min-ns]))]
    (if (< best-cost (* default-cost (- 1.0 (double improvement-threshold))))
      (:strategy best-row)
      default-strategy)))

(defn- measured-selector
  [dispatch runtime-values rows improvement-threshold]
  (let [default-strategy (:default-strategy dispatch)
        choices (mapv (fn [runtime-value]
                        [runtime-value
                         (winner-at (filterv #(= runtime-value (:runtime-value %)) rows)
                                    default-strategy improvement-threshold)])
                      runtime-values)
        ranges (second
                (reduce (fn [[current ranges] [boundary strategy]]
                          (if (= current strategy)
                            [current ranges]
                            [strategy (conj ranges {:at-least boundary :strategy strategy})]))
                        [default-strategy []]
                        choices))]
    {:kind :runtime-scalar-ranges
     :argument (get-in dispatch [:selector :argument])
     :below default-strategy
     :ranges ranges}))

(defn- finite-nonnegative?
  [value]
  (and (number? value)
       (Double/isFinite (double value))
       (not (neg? (double value)))))

(defn- validate-persisted-row!
  [signatures expected-runtime-values row]
  (let [strategy (:strategy row)
        signature (get signatures strategy)
        candidate-hash (:source-hash signature)]
    (when-not signature
      (throw (ex-info "persisted tuning row names an absent strategy"
                      {:reason :dispatch-tuning-evidence-strategy :row row
                       :strategies (set (keys signatures))})))
    (when-not (contains? expected-runtime-values (:runtime-value row))
      (throw (ex-info "persisted tuning row names an unsampled runtime value"
                      {:reason :dispatch-tuning-evidence-runtime-value :row row
                       :runtime-values expected-runtime-values})))
    (if (= :inapplicable (:status row))
      (when-not (and (= candidate-hash (:candidate-hash row))
                     (vector? (:violations row)) (seq (:violations row))
                     (every? #(and (map? %) (keyword? (:reason %))) (:violations row))
                     (not (contains? row :measurement))
                     (not (contains? row :validation)))
        (throw (ex-info "persisted inapplicable tuning row is invalid"
                        {:reason :dispatch-tuning-evidence-inapplicable :row row})))
      (let [measured (:measurement row)
            validation (:validation row)]
        (when-not (and (not (contains? row :status))
                       (map? measured)
                       (every? finite-nonnegative?
                               (map measured [:min-ns :median-ns :p75-ns :mean-ns :cv
                                              :budget-ms :compile-ms]))
                       (true? (:stationary? measured))
                       (integer? (:n measured)) (pos? (long (:n measured)))
                       (integer? (:warmup-iterations measured))
                       (not (neg? (long (:warmup-iterations measured))))
                       (contains? #{:warm :cold} (:cold-warm measured))
                       (= :device-event (:timing-source measured))
                       (map? (:hashes measured)))
          (throw (ex-info "persisted tuning row has invalid measurement evidence"
                          {:reason :dispatch-tuning-evidence-measurement :row row})))
        (when-not (and (map? validation)
                       (true? (:passed? validation))
                       (string? (:oracle-hash validation))
                       (not-empty (:oracle-hash validation))
                       (= candidate-hash (:candidate-hash validation)))
          (throw (ex-info "persisted tuning row has invalid oracle evidence"
                          {:reason :dispatch-tuning-evidence-oracle :row row})))))))

(defn- validate-tuning-evidence!
  [dispatch tuning]
  (let [dispatch (kdispatch/validate! dispatch)
        identity (:identity tuning)
        runtime-values (get-in identity [:policy :runtime-values])
        improvement-threshold (get-in identity [:policy :improvement-threshold])
        fixed? (= :fixed-strategy (get-in dispatch [:selector :kind]))
        expected-runtime-values (if fixed? #{:fixed} (set runtime-values))
        signatures (into {} (map (juxt :strategy clojure.core/identity))
                         (:alternatives identity))
        rows (:measurements tuning)
        expected-pairs (set (for [runtime-value expected-runtime-values
                                  strategy (keys signatures)]
                              [runtime-value strategy]))
        actual-pairs (mapv (juxt :runtime-value :strategy) rows)]
    (when-not (and (vector? runtime-values)
                   (if fixed?
                     (empty? runtime-values)
                     (and (seq runtime-values)
                          (= runtime-values (vec (sort (distinct runtime-values))))
                          (every? #(and (number? %) (Double/isFinite (double %)))
                                  runtime-values)))
                   (number? improvement-threshold)
                   (<= 0.0 (double improvement-threshold))
                   (< (double improvement-threshold) 1.0))
      (throw (ex-info "persisted tuning policy is invalid"
                      {:reason :dispatch-tuning-evidence-policy
                       :policy (:policy identity) :fixed? fixed?})))
    (doseq [row rows]
      (validate-persisted-row! signatures expected-runtime-values row))
    (when-not (and (= expected-pairs (set actual-pairs))
                   (= (count expected-pairs) (count actual-pairs)))
      (throw (ex-info "persisted tuning evidence does not cover every candidate and sample once"
                      {:reason :dispatch-tuning-evidence-coverage
                       :expected expected-pairs :actual actual-pairs})))
    (let [selector (if fixed?
                     {:kind :fixed-strategy
                      :strategy (winner-at rows (:default-strategy dispatch)
                                           improvement-threshold)}
                     (measured-selector dispatch runtime-values rows improvement-threshold))]
      (when-not (= selector (:selector tuning))
        (throw (ex-info "persisted tuning selector does not follow from its measurements"
                        {:reason :dispatch-tuning-evidence-selector
                         :expected selector :actual (:selector tuning)})))
      tuning)))

(defn tune!
  "Measure, validate, and cache a generic KernelDispatch selector OFFLINE.

   `runtime-values` are concrete numeric samples for the dispatch selector argument.
   `benchmark-fn` is called as (benchmark-fn executable runtime-value) and must return:

     {:measurement Measurement
      :validation {:passed? true :oracle-hash string :candidate-hash executable-source-hash ...}}

   or {:status :inapplicable :candidate-hash executable-source-hash :violations [...]} from
   preflight, without a measurement or validation. These rows cannot win; the default must remain
   applicable for every sample. Runtime admission must recheck cached/transported choices.

   Correctness is therefore established before a timing can influence selection. Every result
   that participates in selection must be stationary and device-event timed. A candidate must improve on the dispatch default by
   more than `improvement-threshold` (default 0.1%) at a sample or the default wins that sample.
   Cache hits do not invoke benchmark-fn."
  [dispatch descriptor runtime-values benchmark-fn
   & {:keys [numerical-mode layout improvement-threshold force?]
      :or {improvement-threshold 0.001 force? false}}]
  (let [dispatch (kdispatch/validate! dispatch)
        runtime-values (vec (sort (distinct runtime-values)))]
    (when-not (and (seq runtime-values) (every? #(and (number? %)
                                                      (Double/isFinite (double %)))
                                                runtime-values))
      (throw (ex-info "dispatch tuning runtime values must be non-empty finite numbers"
                      {:runtime-values runtime-values})))
    (when-not (and (number? improvement-threshold)
                   (<= 0.0 (double improvement-threshold))
                   (< (double improvement-threshold) 1.0))
      (throw (ex-info "dispatch improvement-threshold must be in [0,1)"
                      {:improvement-threshold improvement-threshold})))
    (let [identity (tuning-identity dispatch descriptor runtime-values numerical-mode layout
                                    improvement-threshold)
          key (cache-key identity)]
      (or (when-not force? (cache-get dispatch identity))
          (let [rows (mapv (fn [[runtime-value artifact]]
                             (validate-benchmark-result!
                              artifact runtime-value (benchmark-fn artifact runtime-value)))
                           (for [runtime-value runtime-values
                                 artifact (:alternatives dispatch)]
                             [runtime-value artifact]))
                selector (measured-selector dispatch runtime-values rows improvement-threshold)
                _ (kdispatch/with-selector dispatch selector)
                tuning (->DispatchTuning key identity selector rows)]
            (cache-put! tuning))))))

(defn tune-fixed!
  "Measure one static workload across every alternative and cache a fixed winning strategy.

  This is the schedule mode for compile-time axes such as stage depth and shared-memory layout:
  they do not require manufacturing a runtime ABI scalar. The same oracle, stationary device-event,
  source-hash, device, numerical and layout gates used by sampled tuning remain mandatory.
  Source-bound :inapplicable results retain their preflight diagnostics but cannot win; the
  declared default must remain applicable. Runtime binding must recheck transported choices."
  [dispatch descriptor benchmark-fn
   & {:keys [numerical-mode layout improvement-threshold force?]
      :or {improvement-threshold 0.001 force? false}}]
  (let [dispatch (kdispatch/validate! dispatch)]
    (when-not (= :fixed-strategy (get-in dispatch [:selector :kind]))
      (throw (ex-info "fixed dispatch tuning requires a fixed-strategy selector"
                      {:reason :dispatch-tuning-fixed-selector
                       :dispatch-id (:id dispatch) :selector (:selector dispatch)})))
    (when-not (and (number? improvement-threshold)
                   (<= 0.0 (double improvement-threshold))
                   (< (double improvement-threshold) 1.0))
      (throw (ex-info "dispatch improvement-threshold must be in [0,1)"
                      {:improvement-threshold improvement-threshold})))
    (let [identity (tuning-identity dispatch descriptor [] numerical-mode layout
                                    improvement-threshold)
          key (cache-key identity)]
      (or (when-not force? (cache-get dispatch identity))
          (let [rows (mapv (fn [executable]
                             (validate-benchmark-result!
                              executable :fixed (benchmark-fn executable)))
                           (:alternatives dispatch))
                winner (winner-at rows (:default-strategy dispatch) improvement-threshold)
                selector {:kind :fixed-strategy :strategy winner}
                _ (kdispatch/with-selector dispatch selector)
                tuning (->DispatchTuning key identity selector rows)]
            (cache-put! tuning))))))

(defn apply-tuning
  "Bake a tuning selector into a dispatch after rechecking its complete target identity."
  [dispatch tuning descriptor numerical-mode layout]
  (when-not (dispatch-tuning? tuning)
    (throw (ex-info "apply-tuning requires a DispatchTuning" {:tuning tuning})))
  (let [{:keys [runtime-values improvement-threshold]} (:policy (:identity tuning))
        identity (tuning-identity dispatch descriptor runtime-values numerical-mode layout
                                  improvement-threshold)]
    (when-not (= identity (:identity tuning))
      (throw (ex-info "dispatch tuning identity differs from the target dispatch"
                      {:expected identity :actual (:identity tuning)})))
    (kdispatch/with-selector dispatch (:selector tuning))))
