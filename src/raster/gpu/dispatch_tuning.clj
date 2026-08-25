(ns raster.gpu.dispatch-tuning
  "Explicit offline autotuning for a KernelDispatch.

   Compilation and runtime selection never benchmark. This namespace measures already-emitted,
   ABI-compatible executable alternatives through a caller-supplied validated benchmark, converts
   winners at sampled runtime scalar values into a piecewise selector, and atomically caches it.
   Applying a result rechecks the full identity: device, emitted sources, ABI, numerical mode, and
   layout. A stale or cross-device result therefore cannot silently select a kernel."
  (:require [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.charset StandardCharsets]
           [java.security MessageDigest]))

(def tuning-version 2)

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
                                    :artifact (select-keys (:operation node)
                                                           [:kernel-name :target :source :abi
                                                            :arguments :launch :temporaries])})
                                 (:nodes executable))}
                   (select-keys executable
                                [:kernel-name :target :source :launch :temporaries]))]
    {:kind (kexec/kind executable)
     :strategy (kdispatch/alternative-strategy executable)
     :target (kexec/target executable)
     :entry-points (kexec/entry-points executable)
     :source-hash (sha256 (pr-str (canonical-data schedule)))
     :abi-hash (sha256 (pr-str (canonical-data (kexec/abi executable))))
     :arguments-hash (sha256 (pr-str (kexec/arguments executable)))
     :effects-hash (sha256 (pr-str (canonical-data (kexec/effects executable))))}))

(defn- device-signature
  [descriptor]
  (select-keys descriptor
               [:device-id :device-name :vendor :arch :driver-version
                :machine-lanes :grf-bytes-per-lane :bandwidth-bytes-s :peak-flops
                :subgroup-size :max-workgroup-size :matrix]))

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
      :device (device-signature descriptor)
      :numerical-mode numerical-mode
      :layout layout
      :policy {:runtime-values (vec runtime-values)
               :improvement-threshold (double improvement-threshold)}
      :alternatives (mapv executable-signature (:alternatives dispatch))})))

(defn cache-key
  [identity]
  (sha256 (pr-str (canonical-data identity))))

(defn- tuning-from-data
  [dispatch identity key data]
  (when (and (= tuning-version (:version data))
             (= key (:key data))
             (= identity (:identity data))
             (map? (:selector data))
             (vector? (:measurements data)))
    (kdispatch/with-selector dispatch (:selector data))
    (->DispatchTuning key identity (:selector data) (:measurements data))))

(defn cache-get
  "Read and validate a cached result for the exact dispatch identity. Corrupt/stale entries miss."
  [dispatch identity]
  (let [key (cache-key identity)]
    (try
      (tuning-from-data dispatch identity key (cache/read-entry key))
      (catch Exception _ nil))))

(defn cache-put!
  "Atomically persist a DispatchTuning as plain EDN."
  [tuning]
  (when-not (dispatch-tuning? tuning)
    (throw (ex-info "dispatch tuning cache requires a DispatchTuning" {:value tuning})))
  (let [data {:version tuning-version
              :key (:key tuning)
              :identity (:identity tuning)
              :selector (:selector tuning)
              :measurements (:measurements tuning)}]
    (cache/write-entry! (:key tuning) data)
    tuning))

(def ^:private measurement-fields
  [:min-ns :median-ns :p75-ns :mean-ns :cv :stationary? :n :warmup-iterations
   :budget-ms :cold-warm :timing-source :compile-ms :hashes])

(defn- validate-benchmark-result!
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

(defn- winner-at
  [rows default-strategy improvement-threshold]
  (let [by-strategy (into {} (map (juxt :strategy identity)) rows)
        default-row (or (get by-strategy default-strategy)
                        (throw (ex-info "dispatch measurements omit the default strategy"
                                        {:default-strategy default-strategy
                                         :strategies (set (keys by-strategy))})))
        best-row (apply min-key #(get-in % [:measurement :min-ns]) rows)
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

(defn tune!
  "Measure, validate, and cache a generic KernelDispatch selector OFFLINE.

   `runtime-values` are concrete numeric samples for the dispatch selector argument.
   `benchmark-fn` is called as (benchmark-fn executable runtime-value) and must return:

     {:measurement Measurement
      :validation {:passed? true :oracle-hash string :candidate-hash executable-source-hash ...}}

   Correctness is therefore established before a timing can influence selection. Every result
   must be stationary and device-event timed. A candidate must improve on the dispatch default by
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
