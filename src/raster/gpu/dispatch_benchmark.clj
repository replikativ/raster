(ns raster.gpu.dispatch-benchmark
  "Concrete resident execution driver for generic KernelDispatch autotuning.

   This namespace contains no operation recognition or schedule policy. A dispatch already owns
   emitted ABI-compatible alternatives; a benchmark case supplies resident arguments, an
   independent correctness oracle, and (only when necessary) a reset action. Every candidate is
   validated before it is measured, device time comes from profiling events, and DispatchTuning
   remains responsible for selection and caching."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]))

(def ^:private reserved-measurement-options
  #{:before-sample! :compile-ms :hashes :timing-source})

(defn- runtime-number
  [value]
  (if (and (map? value) (contains? value :value)) (:value value) value))

(defn- stateful-artifact?
  [artifact]
  (let [{:keys [reads writes]} (:effects artifact)]
    (boolean (seq (set/intersection (set reads) (set writes))))))

(defn- validate-case!
  [dispatch artifact runtime-value case]
  (when-not (map? case)
    (throw (ex-info "dispatch benchmark case must be a map"
                    {:runtime-value runtime-value :case case})))
  (let [{:keys [arguments validate! before-run! measurement]} case
        selector-argument (get-in dispatch [:selector :argument])
        indexes (keep-indexed (fn [index value]
                                (when (= selector-argument value) index))
                              (:arguments artifact))]
    (when-not (vector? arguments)
      (throw (ex-info "dispatch benchmark case arguments must be an ABI-ordered vector"
                      {:runtime-value runtime-value :arguments arguments})))
    (when-not (= (count (:abi artifact)) (count arguments))
      (throw (ex-info "dispatch benchmark case argument count differs from the artifact ABI"
                      {:runtime-value runtime-value
                       :expected (count (:abi artifact)) :actual (count arguments)})))
    (when-not (= 1 (count indexes))
      (throw (ex-info "dispatch benchmark selector must have one artifact argument position"
                      {:runtime-value runtime-value :argument selector-argument
                       :indexes (vec indexes)})))
    (let [actual (runtime-number (nth arguments (first indexes)))]
      (when-not (= runtime-value actual)
        (throw (ex-info "dispatch benchmark case does not bind its sampled runtime value"
                        {:selector-argument selector-argument
                         :sampled runtime-value :bound actual}))))
    (when-not (ifn? validate!)
      (throw (ex-info "dispatch benchmark case requires a :validate! oracle callback"
                      {:runtime-value runtime-value})))
    (when-not (or (nil? before-run!) (ifn? before-run!))
      (throw (ex-info "dispatch benchmark :before-run! must be callable"
                      {:runtime-value runtime-value :before-run! before-run!})))
    (when (and (stateful-artifact? artifact) (nil? before-run!))
      (throw (ex-info "stateful dispatch candidates require :before-run! restoration"
                      {:runtime-value runtime-value :effects (:effects artifact)})))
    (when-not (or (nil? measurement) (map? measurement))
      (throw (ex-info "dispatch benchmark :measurement options must be a map"
                      {:runtime-value runtime-value :measurement measurement})))
    (when-let [reserved (seq (set/intersection reserved-measurement-options
                                               (set (keys measurement))))]
      (throw (ex-info "dispatch benchmark measurement options contain driver-owned fields"
                      {:runtime-value runtime-value :reserved (set reserved)})))
    case))

(defn- validate-oracle-result!
  [artifact runtime-value validation]
  (let [source-hash (:source-hash (tuning/artifact-signature artifact))]
    (when-not (and (map? validation)
                   (true? (:passed? validation))
                   (string? (:oracle-hash validation))
                   (not-empty (:oracle-hash validation)))
      (throw (ex-info "dispatch candidate failed its oracle before measurement"
                      {:runtime-value runtime-value
                       :strategy (kdispatch/artifact-strategy artifact)
                       :validation validation})))
    (assoc validation :candidate-hash source-hash)))

(defn benchmark-candidate!
  "Validate and device-time one emitted dispatch alternative at one sampled runtime value.

   `case-fn` is called as `(case-fn runtime-value)` and returns:

     {:arguments   [buffer-key-or-view ... typed-scalars ...] ; physical ABI order
      :validate!   (fn [context] {:passed? true :oracle-hash string ...})
      :before-run! (fn [context] ...)                        ; optional restore
      :measurement {:budget-ms ... :min-samples ...}}        ; optional

   The context contains :session, :dispatch, :artifact, :runtime-value, :case, and after the
   validation replay, :outputs. The driver supplies :candidate-hash itself, so a callback cannot
   accidentally validate one source and bless another."
  [session dispatch artifact runtime-value case-fn & {:keys [measurement]}]
  (let [dispatch (kdispatch/validate! dispatch)
        strategy (kdispatch/artifact-strategy artifact)
        expected (kdispatch/artifact dispatch strategy)
        _ (when-not (= expected artifact)
            (throw (ex-info "dispatch benchmark artifact is not the registered alternative"
                            {:strategy strategy :expected expected :artifact artifact})))
        _ (when-not (or (nil? measurement) (map? measurement))
            (throw (ex-info "dispatch benchmark global :measurement options must be a map"
                            {:measurement measurement})))
        case (validate-case! dispatch artifact runtime-value (case-fn runtime-value))
        measurement-options (merge measurement (:measurement case))
        reserved (set/intersection reserved-measurement-options
                                   (set (keys measurement-options)))]
    (when (seq reserved)
      (throw (ex-info "dispatch benchmark measurement options contain driver-owned fields"
                      {:runtime-value runtime-value :reserved reserved})))
    (let [key [::candidate runtime-value (kdispatch/artifact-strategy artifact) (random-uuid)]
          started (System/nanoTime)
          handle (gpu/bind-kernel-call! session key artifact (:arguments case)
                                        {:profile? true
                                         :group-count (:group-count case)})
          compile-ms (/ (- (System/nanoTime) started) 1.0e6)
          base-context {:session session :dispatch dispatch :artifact artifact
                        :runtime-value runtime-value :case case}]
      (try
        (when-let [before-run! (:before-run! case)] (before-run! base-context))
        (let [outputs (gpu/run-kernel-graph! session handle)
              validation (validate-oracle-result!
                          artifact runtime-value
                          ((:validate! case) (assoc base-context :outputs outputs)))
              before-sample! (when-let [before-run! (:before-run! case)]
                               #(before-run! base-context))
              hashes {:candidate (:candidate-hash validation)
                      :oracle (:oracle-hash validation)}
              opts (cond-> measurement-options
                     before-sample! (assoc :before-sample! before-sample!)
                     true (assoc :compile-ms compile-ms :hashes hashes))
              measured (apply gpu/measure-bound-kernel-graph!
                              session handle (mapcat identity opts))]
          {:measurement measured :validation validation})
        (finally
          (gpu/release-kernel-graph! session handle))))))

(defn tune-dispatch!
  "Tune an emitted KernelDispatch through resident validation and device-event measurement.

   This is an explicit offline action. Compilation and ordinary dispatch remain pure and never
   acquire a benchmark side effect. Options are DispatchTuning identity/policy fields plus a
   global :measurement option map; an individual benchmark case may refine those sampling bounds."
  [session dispatch descriptor runtime-values case-fn
   & {:keys [numerical-mode layout improvement-threshold force? measurement]
      :or {improvement-threshold 0.001 force? false}}]
  (tuning/tune!
   dispatch descriptor runtime-values
   (fn [artifact runtime-value]
     (benchmark-candidate! session dispatch artifact runtime-value case-fn
                           :measurement measurement))
   :numerical-mode numerical-mode
   :layout layout
   :improvement-threshold improvement-threshold
   :force? force?))
