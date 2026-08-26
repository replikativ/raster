(ns raster.gpu.dispatch-benchmark
  "Concrete resident execution driver for generic KernelDispatch autotuning.

   This namespace contains no operation recognition or schedule policy. A dispatch already owns
   emitted ABI-compatible executable alternatives; a benchmark case supplies resident arguments, an
   independent correctness oracle, and (only when necessary) a reset action. Every candidate is
   validated before it is measured, device time comes from profiling events, and DispatchTuning
   remains responsible for selection and caching."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.link :as link]
            [raster.gpu.program-tuning :as program-tuning]))

(def ^:private reserved-measurement-options
  #{:before-sample! :compile-ms :hashes :timing-source})

(defn- runtime-number
  [value]
  (if (and (map? value) (contains? value :value)) (:value value) value))

(defn- stateful-executable?
  [executable]
  (let [{:keys [reads writes]} (kexec/effects executable)]
    (boolean (seq (set/intersection (set reads) (set writes))))))

(defn- validate-case!
  [dispatch executable runtime-value case]
  (when-not (map? case)
    (throw (ex-info "dispatch benchmark case must be a map"
                    {:runtime-value runtime-value :case case})))
  (let [{:keys [arguments validate! before-run! measurement]} case
        selector-argument (get-in dispatch [:selector :argument])
        indexes (keep-indexed (fn [index value]
                                (when (= selector-argument value) index))
                              (kexec/arguments executable))]
    (when-not (vector? arguments)
      (throw (ex-info "dispatch benchmark case arguments must be an ABI-ordered vector"
                      {:runtime-value runtime-value :arguments arguments})))
    (when-not (= (count (kexec/abi executable)) (count arguments))
      (throw (ex-info "dispatch benchmark case argument count differs from the executable ABI"
                      {:runtime-value runtime-value
                       :expected (count (kexec/abi executable)) :actual (count arguments)})))
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
    (when (and (stateful-executable? executable) (nil? before-run!))
      (throw (ex-info "stateful dispatch candidates require :before-run! restoration"
                      {:runtime-value runtime-value :effects (kexec/effects executable)})))
    (when-not (or (nil? measurement) (map? measurement))
      (throw (ex-info "dispatch benchmark :measurement options must be a map"
                      {:runtime-value runtime-value :measurement measurement})))
    (when-let [reserved (seq (set/intersection reserved-measurement-options
                                               (set (keys measurement))))]
      (throw (ex-info "dispatch benchmark measurement options contain driver-owned fields"
                      {:runtime-value runtime-value :reserved (set reserved)})))
    case))

(defn- validate-oracle-result!
  [executable runtime-value validation]
  (let [source-hash (:source-hash (tuning/executable-signature executable))]
    (when-not (and (map? validation)
                   (true? (:passed? validation))
                   (string? (:oracle-hash validation))
                   (not-empty (:oracle-hash validation)))
      (throw (ex-info "dispatch candidate failed its oracle before measurement"
                      {:runtime-value runtime-value
                       :strategy (kdispatch/alternative-strategy executable)
                       :validation validation})))
    (assoc validation :candidate-hash source-hash)))

(defn benchmark-candidate!
  "Validate and device-time one emitted dispatch alternative at one sampled runtime value.

   `case-fn` is called as `(case-fn runtime-value)` and returns:

     {:arguments   [buffer-key-or-view ... typed-scalars ...] ; physical ABI order
      :validate!   (fn [context] {:passed? true :oracle-hash string ...})
      :before-run! (fn [context] ...)                        ; optional restore
      :measurement {:budget-ms ... :min-samples ...}}        ; optional

   The context contains :session, :dispatch, :executable, :runtime-value, :case, and after the
   validation replay, :outputs. The driver supplies :candidate-hash itself, so a callback cannot
   accidentally validate one source and bless another."
  [session dispatch executable runtime-value case-fn & {:keys [measurement]}]
  (let [dispatch (kdispatch/validate! dispatch)
        strategy (kdispatch/alternative-strategy executable)
        expected (kdispatch/alternative dispatch strategy)
        _ (when-not (= expected executable)
            (throw (ex-info "dispatch benchmark executable is not the registered alternative"
                            {:strategy strategy :expected expected :executable executable})))
        _ (when-not (or (nil? measurement) (map? measurement))
            (throw (ex-info "dispatch benchmark global :measurement options must be a map"
                            {:measurement measurement})))
        case (validate-case! dispatch executable runtime-value (case-fn runtime-value))
        measurement-options (merge measurement (:measurement case))
        reserved (set/intersection reserved-measurement-options
                                   (set (keys measurement-options)))]
    (when (seq reserved)
      (throw (ex-info "dispatch benchmark measurement options contain driver-owned fields"
                      {:runtime-value runtime-value :reserved reserved})))
    (let [key [::candidate runtime-value (kdispatch/alternative-strategy executable) (random-uuid)]
          started (System/nanoTime)
          handle (gpu/bind-kernel-executable! session key executable (:arguments case)
                                              {:profile? true
                                               :group-count (:group-count case)})
          compile-ms (/ (- (System/nanoTime) started) 1.0e6)
          base-context {:session session :dispatch dispatch :executable executable
                        :runtime-value runtime-value :case case}]
      (try
        (when-let [before-run! (:before-run! case)] (before-run! base-context))
        (let [outputs (gpu/run-kernel-graph! session handle)
              validation (validate-oracle-result!
                          executable runtime-value
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
   (fn [executable runtime-value]
     (benchmark-candidate! session dispatch executable runtime-value case-fn
                           :measurement measurement))
   :numerical-mode numerical-mode
   :layout layout
   :improvement-threshold improvement-threshold
   :force? force?))

(defn tuning-schedule-override
  "Turn a measured selector into the compiler schedule override declared by its dispatch.

   Emitters own the path because only they know which schedule axis produced their compatible
   alternatives. This function is generic: it validates the selector against the dispatch and
   writes no operation-specific key itself."
  [dispatch dispatch-tuning descriptor numerical-mode layout]
  (let [dispatch (kdispatch/validate! dispatch)
        _ (when-not (tuning/dispatch-tuning? dispatch-tuning)
            (throw (ex-info "schedule override requires a DispatchTuning"
                            {:tuning dispatch-tuning})))
        path (get-in dispatch [:attributes :tuning :schedule-path])
        schedule-key (get-in dispatch [:attributes :tuning :schedule-key])
        target-path (cond-> path (some? schedule-key) (conj schedule-key))
        selector (:selector dispatch-tuning)]
    (when-not (and (vector? path) (seq path) (every? keyword? path))
      (throw (ex-info "emitted dispatch does not declare a tuning schedule path"
                      {:dispatch-id (:id dispatch) :schedule-path path})))
    (when-not (or (nil? schedule-key)
                  (and (string? schedule-key) (not-empty schedule-key))
                  (keyword? schedule-key))
      (throw (ex-info "emitted dispatch declares an invalid tuning schedule key"
                      {:dispatch-id (:id dispatch) :schedule-key schedule-key})))
    ;; Recheck the full device/artifact/ABI/numerical/layout identity before admitting cached or
    ;; transported tuning data into recompilation.
    (tuning/apply-tuning dispatch dispatch-tuning descriptor numerical-mode layout)
    (assoc-in {} target-path selector)))

(defn- linked-case-fn
  [executable instance-selector step-selector case-fn]
  (fn [runtime-value]
    (let [case (case-fn runtime-value)]
      (when-not (map? case)
        (throw (ex-info "linked dispatch benchmark case must be a map"
                        {:runtime-value runtime-value :case case})))
      (when-not (contains? case :descriptor-arguments)
        (throw (ex-info "linked dispatch benchmark case requires :descriptor-arguments"
                        {:runtime-value runtime-value :case-keys (set (keys case))})))
      (let [binding (link/dispatch-arguments
                     executable instance-selector step-selector (:descriptor-arguments case))]
        (assoc case
               :arguments (:arguments binding)
               ;; Validation callbacks reach the compiler-owned reference plan and the projected
               ;; host buffer/scalar environment through (:linked-binding (:case context)).
               :linked-binding (dissoc binding :arguments))))))

(defn tune-linked-dispatch!
  "Tune one KernelDispatch embedded in a live LinkedExecutable.

   This closes the descriptor-to-driver seam without teaching the runtime any operation. Each
   `case-fn` result supplies `:descriptor-arguments` in descriptor order plus :validate!,
   :before-run!, and :measurement callbacks accepted by tune-dispatch!. The bridge projects the
   step's ordered :argument-specs onto certified resident node views and typed scalar values. When
   :instance is omitted, the executable must contain exactly one instance; when :step is omitted,
   that instance must contain exactly one dispatch step.

   Numerical mode and layout default to emitter-owned tuning metadata but may be supplied
   explicitly. Returns the DispatchTuning together with a schedule override ready to pass as
   `:schedule` to compile-gpu-program. Compilation remains pure; this function is explicitly an
   offline action."
  [executable descriptor runtime-values case-fn
   & {:keys [instance step numerical-mode layout improvement-threshold force? measurement]
      :or {improvement-threshold 0.001 force? false}}]
  (let [{:keys [dispatch step-index instance-id] compiled-step :step}
        (link/linked-dispatch executable instance step)
        contract (get-in dispatch [:attributes :tuning])
        numerical-mode (or numerical-mode (:numerical-mode contract))
        layout (or layout (:layout contract))
        result (tune-dispatch!
                (:session executable) dispatch descriptor runtime-values
                (linked-case-fn executable instance step case-fn)
                :numerical-mode numerical-mode
                :layout layout
                :improvement-threshold improvement-threshold
                :force? force?
                :measurement measurement)]
    {:tuning result
     :selector (:selector result)
     :schedule-override (tuning-schedule-override dispatch result descriptor
                                                  numerical-mode layout)
     :instance-id instance-id
     :step-index step-index
     :phase (:phase compiled-step)}))

(defn tune-linked-dispatches!
  "Execute an explicit program tuning plan and return one collision-free schedule override.

   `runtime-values-fn` receives a manifest group. `case-fn` receives that group and one runtime
   value, and returns the compiled benchmark case accepted by tune-linked-dispatch!. Equivalent
   sites are measured once through their first descriptor step. `:max-measurements` is an optional
   fail-before-execution upper bound on alternatives × distinct runtime samples across selected
   groups; cached results may perform fewer physical measurements."
  [executable descriptor plan runtime-values-fn case-fn
   & {:keys [instance max-measurements improvement-threshold force? measurement]
      :or {improvement-threshold 0.001 force? false}}]
  (let [plan (program-tuning/validate-plan! plan)]
    (when-not (ifn? runtime-values-fn)
      (throw (ex-info "program tuning requires a runtime-values callback"
                      {:reason :invalid-program-tuning-runtime-values-callback})))
    (when-not (ifn? case-fn)
      (throw (ex-info "program tuning requires a benchmark case callback"
                      {:reason :invalid-program-tuning-case-callback})))
    (when-not (or (nil? max-measurements)
                  (and (integer? max-measurements)
                       (not (neg? (long max-measurements)))))
      (throw (ex-info "program tuning :max-measurements must be a non-negative integer"
                      {:reason :invalid-program-tuning-measurement-budget
                       :max-measurements max-measurements})))
    (let [jobs
          (mapv
           (fn [group]
             (let [step (:representative-step-index group)
                   {:keys [dispatch]} (link/linked-dispatch executable instance step)
                   actual-signature (program-tuning/dispatch-signature dispatch)
                   supplied-runtime-values (runtime-values-fn group)
                   _runtime-values
                   (when-not (and (coll? supplied-runtime-values)
                                  (seq supplied-runtime-values)
                                  (every? #(and (number? %)
                                                (Double/isFinite (double %)))
                                          supplied-runtime-values))
                     (throw (ex-info "program tuning group requires non-empty finite runtime values"
                                     {:reason :invalid-program-tuning-runtime-values
                                      :group-id (:id group)
                                      :runtime-values supplied-runtime-values})))
                   runtime-values (vec (sort (distinct supplied-runtime-values)))]
               (when-not (= (:signature group) actual-signature)
                 (throw (ex-info "program tuning plan does not match its bound program step"
                                 {:reason :program-tuning-plan-program-mismatch
                                  :group-id (:id group) :step-index step
                                  :planned (:signature group) :actual actual-signature})))
               {:group group
                :runtime-values runtime-values
                :planned-measurements
                (* (count (get-in group [:signature :alternatives]))
                   (count runtime-values))}))
           (:selected-groups plan))
          planned-measurements (reduce + 0 (map :planned-measurements jobs))]
      (when (and (some? max-measurements)
                 (> planned-measurements (long max-measurements)))
        (throw (ex-info "program tuning plan exceeds its physical measurement budget"
                        {:reason :program-tuning-measurement-budget-exceeded
                         :planned-measurements planned-measurements
                         :max-measurements max-measurements
                         :groups (mapv (comp :id :group) jobs)})))
      (let [results
            (mapv
             (fn [{:keys [group runtime-values planned-measurements]}]
               (assoc
                (tune-linked-dispatch!
                 executable descriptor runtime-values
                 #(case-fn group %)
                 :instance instance
                 :step (:representative-step-index group)
                 :improvement-threshold improvement-threshold
                 :force? force?
                 :measurement measurement)
                :group-id (:id group)
                :site-count (:site-count group)
                :planned-measurements planned-measurements))
             jobs)]
        {:plan plan
         :planned-measurements planned-measurements
         :results results
         :schedule-override
         (program-tuning/merge-schedule-overrides (mapv :schedule-override results))}))))
