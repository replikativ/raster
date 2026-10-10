(ns raster.compiler.ir.emitted-parallel-program-call
  "Pure runtime binding for one equation-first emitted parallel program.

   Construction evaluates only effect-free host scalar equations that are independent of device
   results. Every numerical equation is then bound from its emitted ABI and retained physical
   result contracts. The resulting call owns no driver handles and never reads retained source."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(defrecord EvaluatedHostEquation [equation operands results])
(defrecord EmittedEquationCall [equation graph buffers scalar-values outputs])
(defrecord EmittedParallelProgramCall
           [program steps entry-buffers buffers scalar-values loop-scratch outputs attributes])

(defn- record-kind?
  [record-class value]
  (and value (= record-class (.getName (class value)))))

(defn evaluated-host-equation?
  [value]
  (record-kind?
   "raster.compiler.ir.emitted_parallel_program_call.EvaluatedHostEquation" value))

(defn emitted-equation-call?
  [value]
  (record-kind?
   "raster.compiler.ir.emitted_parallel_program_call.EmittedEquationCall" value))

(defn emitted-program-call?
  [value]
  (record-kind?
   "raster.compiler.ir.emitted_parallel_program_call.EmittedParallelProgramCall" value))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :emitted-parallel-program-call))))

(defn- typed-scalar?
  [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn- checked-scalar
  [values id value]
  (let [expected (:dtype (get values id))]
    (when-not (typed-scalar? value)
      (fail! :emitted-program-scalar
             "parallel program scalars require explicit runtime dtypes"
             {:value id :scalar value}))
    (when-not (= (dtype/canon expected) (dtype/canon (:type value)))
      (fail! :emitted-program-scalar-type
             "runtime scalar dtype differs from its retained AbstractValue"
             {:value id :expected expected :actual (:type value)}))
    value))

(defn- require-buffer
  [buffers id role]
  (let [value (get buffers id ::missing)]
    (when (or (= ::missing value) (nil? value))
      (fail! :emitted-program-buffer
             "parallel program buffer binding is missing"
             {:value id :role role}))
    value))

(defn- validate-host-step!
  [values step]
  (when-not (evaluated-host-equation? step)
    (fail! :emitted-program-host-step "expected an EvaluatedHostEquation" {:step step}))
  (let [{:keys [equation operands results]} step]
    (when-not (and (program/equation? equation)
                   (true? (get-in equation [:attributes :host-only]))
                   (empty? (:operations equation)))
      (fail! :emitted-program-host-equation
             "evaluated host step does not retain a host-only equation"
             {:equation equation}))
    (when-not (= (set (:operands equation)) (set (keys operands)))
      (fail! :emitted-program-host-operands
             "evaluated host operands differ from the retained equation"
             {:equation (:id equation) :expected (:operands equation)
              :actual (keys operands)}))
    (when-not (= (set (:results equation)) (set (keys results)))
      (fail! :emitted-program-host-results
             "evaluated host results differ from the retained equation"
             {:equation (:id equation) :expected (:results equation)
              :actual (keys results)}))
    (doseq [[id value] results]
      (checked-scalar values id value))
    step))

(declare validate-result-views!)

(defn- preflight-violations
  [graph scalar-values]
  (try
    (graph-call/preflight! graph scalar-values)
    []
    (catch clojure.lang.ExceptionInfo exception
      (let [reason (:reason (ex-data exception))]
        ;; Only a failed, data-dependent precondition is an inapplicable schedule. Malformed
        ;; certificates, scalar bindings and internal errors must propagate, not trigger fallback.
        (if (= :kernel-precondition-failed reason)
          [{:reason reason}]
          (throw exception))))))

(defn- validate-equation-call-against-boundary!
  ;; Only synchronous callers below supply the exact boundary and storage projection just
  ;; produced by emitted-equation validation. No proof escapes into the call record.
  [call emitted physical candidates]
  (when-not (emitted-equation-call? call)
    (fail! :emitted-program-equation-call "expected an EmittedEquationCall" {:call call}))
  (let [{equation :equation call-graph :graph buffers :buffers
         scalar-values :scalar-values outputs :outputs} call
        expected-graphs (mapv :graph candidates)
        graph call-graph
        runtime-arguments
        (mapv (fn [slot argument]
                (if (= :scalar (:kind slot))
                  (get scalar-values argument ::missing)
                  (get buffers argument ::missing)))
              (:abi graph) (:arguments graph))]
    (when (some #{::missing} runtime-arguments)
      (fail! :emitted-program-equation-interface
             "emitted equation call does not bind its complete graph ABI"
             {:equation (:id equation)}))
    (let [bindings (executable/graph-bindings graph runtime-arguments)]
      (when-not (= {:buffers buffers :scalar-values scalar-values} bindings)
        (fail! :emitted-program-equation-bindings
               "emitted equation call bindings differ from its ordered ABI"
               {:equation (:id equation)})))
    ;; Shape/representation obligations are decidable from bound scalars, before any caller
    ;; opens a session or allocates result storage. Runtime binding repeats this check.
    (graph-call/preflight! graph scalar-values)
    (when-not (= (set (:results equation)) (set (keys outputs)))
      (fail! :emitted-program-equation-outputs
             "emitted equation outputs differ from its logical results"
             {:equation (:id equation) :expected (:results equation)
              :actual (keys outputs)}))
    (when-not (some #(or (identical? % call-graph)
                        (semantic-fingerprint/equivalent? % call-graph))
                    expected-graphs)
      (fail! :emitted-program-equation-graph
             "emitted equation call graph is not a certified alternative"
             {:equation (:id equation)}))
    (validate-result-views! equation emitted physical (or (:result-views call) {}))
    call))

(def ^:dynamic ^:private *validated-boundary-projections* nil)
(def ^:dynamic ^:private *validated-projection-policy* nil)

(defn ^:no-doc without-validation-context
  "Invoke runtime/user code without inheriting synchronous compiler projection authority."
  [invoke]
  (binding [*validated-boundary-projections* nil
            *validated-projection-policy* nil]
    (invoke)))

(defn- same-projection-request? [caller-options]
  (= (numerics/validate-scalar-math-policy! *validated-projection-policy*)
     (numerics/validate-scalar-math-policy! (:scalar-math caller-options))))

(defn- checked-operation-projection [operation caller-options]
  (if *validated-boundary-projections*
    (do
      (when-not (same-projection-request? caller-options)
        (fail! :emitted-program-projection-math-request
               "checked equation projections belong to a different caller math request" {}))
      (emitted-program/operation-projection *validated-boundary-projections* operation))
    (if (equation-dispatch/emitted-equation-dispatch? operation)
      (if (nil? caller-options)
        (equation-dispatch/validate-with-boundary operation)
        (equation-dispatch/validate-with-boundary operation caller-options))
      (let [projection (if (nil? caller-options)
                         (emitted-equation/validate-with-physical-results operation)
                         (emitted-equation/validate-with-physical-results operation caller-options))]
        (assoc projection :candidates [(:boundary projection)])))))

(defn- validate-equation-call-for-request!
  [call caller-options]
  (when-not (emitted-equation-call? call)
    (fail! :emitted-program-equation-call "expected an EmittedEquationCall" {:call call}))
  (let [{:keys [boundary physical-results candidates]}
        (checked-operation-projection (first (:operations (:equation call))) caller-options)]
        ;; Independent outside checked construction. Inside it, only the exact already-checked
        ;; immutable boundary's projection may be reused; every call/binding check still runs.
    (validate-equation-call-against-boundary! call boundary physical-results candidates)))

(defn validate-equation-call!
  ([call] (validate-equation-call-for-request! call nil))
  ([call caller-options] (validate-equation-call-for-request! call caller-options)))

(defn- validate-result-views!
  [equation boundary physical result-views]
  (when-not (map? result-views)
    (fail! :emitted-program-result-views "result views must be a map" {:result-views result-views}))
  (let [algorithm (:algorithm boundary)]
    (doseq [[result destination] result-views]
      (let [producer (when (soac/program-form? algorithm)
                       (some #(when (some #{result} (nth % 2)) %) (soac/equations algorithm)))]
        (when-not (and (some #{result} (:results equation))
                       (= destination (get physical result))
                       producer
                       (contains? '#{contract map} (soac/operation-kind producer)))
          (fail! :emitted-program-result-view
                 "result view must retain a prefix-producing physical storage relation"
                 {:equation (:id equation) :result result :destination destination}))))
    result-views))

(defn- prepare-equation-bindings
  [equation values buffers scalars result-views projections]
  (let [operation (first (:operations equation))
        projection (emitted-program/operation-projection projections operation)
        emitted (:boundary projection)
        ;; The enclosing program supplied exact-operation facts it just checked. No constructor
        ;; context is bound around the host-evaluator callback preceding this preparation phase.
        result-storage (:physical-results projection)
        common-graph (:graph emitted)
        result-views (validate-result-views! equation emitted result-storage
                                           (select-keys result-views (:results equation)))
        buffers
        (reduce
         (fn [bindings result]
           (let [physical (get result-storage result)
                 physical-binding (get bindings physical ::missing)
                 logical-binding (get bindings result ::missing)
                 selected (cond
                            (not= ::missing physical-binding) physical-binding
                            (not= ::missing logical-binding) logical-binding
                            :else (fail! :emitted-program-result-buffer
                                         "numerical result requires preallocated resident storage"
                                         {:equation (:id equation) :result result
                                          :physical-result physical}))]
             (when (and (not= ::missing physical-binding)
                        (not= ::missing logical-binding)
                        (not= physical-binding logical-binding)
                        (not= physical (get result-views result)))
               (fail! :emitted-program-result-alias
                      "logical and physical result bindings disagree"
                      {:equation (:id equation) :result result :physical-result physical}))
             (assoc bindings physical selected result
                    (if (contains? result-views result)
                      (require-buffer bindings result :result-view)
                      selected))))
         buffers (:results equation))
        runtime-arguments
        (mapv (fn [slot argument]
                (if (= :scalar (:kind slot))
                  (let [value (checked-scalar values argument (get scalars argument))]
                    (when-not (= (dtype/canon (:dtype slot))
                                 (dtype/canon (:type value)))
                      (fail! :emitted-program-scalar-abi
                             "runtime scalar dtype differs from the logical graph ABI"
                             {:equation (:id equation) :value argument
                              :expected (:dtype slot) :actual (:type value)}))
                    value)
                  (require-buffer buffers argument :equation-interface)))
              (:abi common-graph) (:arguments common-graph))
        runtime-arguments (executable/typed-runtime-arguments common-graph runtime-arguments)
        bindings (executable/graph-bindings common-graph runtime-arguments)]
    {:bindings (assoc bindings :outputs (select-keys buffers (:results equation)))
     :buffers buffers :boundary emitted :common-graph common-graph
     :runtime-arguments runtime-arguments :result-storage result-storage
     :candidates (:candidates projection)
     :result-views result-views}))

(defn- prepare-equation-call
  [equation values buffers scalars result-views projections]
  (let [{:keys [bindings buffers boundary common-graph runtime-arguments result-storage result-views candidates]}
        (prepare-equation-bindings equation values buffers scalars result-views projections)
        operation (first (:operations equation))
        graph (if (equation-dispatch/emitted-equation-dispatch? operation)
                (:executable
                 (dispatch/admit-alternative
                  (:dispatch operation) runtime-arguments
                  #(preflight-violations % (:scalar-values bindings))))
                common-graph)
        outputs (:outputs bindings)]
    {:call (validate-equation-call-against-boundary!
            (assoc (->EmittedEquationCall equation graph (:buffers bindings)
                                         (:scalar-values bindings) outputs)
                   :result-views result-views)
            boundary result-storage candidates)
     :buffers buffers}))

(defn- evaluate-host-equations
  [parallel-program buffers scalars evaluate-host]
  (let [values (:values parallel-program)]
    (reduce
     (fn [{:keys [scalars device-results host-steps] :as state} equation]
       (if-not (true? (get-in equation [:attributes :host-only]))
         (update state :device-results into (:results equation))
         (do
           (when (seq (:effects equation))
             (fail! :emitted-program-host-effects
                    "host scalar equations must be effect-free before they can be hoisted"
                    {:equation (:id equation) :effects (:effects equation)}))
           (let [device-dependencies (set/intersection device-results
                                                       (set (:operands equation)))]
             (when (seq device-dependencies)
               (fail! :emitted-program-host-device-dependency
                      "a host scalar equation cannot be staged before a device result"
                      {:equation (:id equation) :values device-dependencies})))
           (when-not (ifn? evaluate-host)
             (fail! :emitted-program-host-evaluator
                    "an emitted program with host scalar equations requires an evaluator"
                    {:equation (:id equation)}))
           (let [operand-bindings
                 (into {}
                       (map (fn [id]
                              [id (cond
                                    (contains? scalars id) (checked-scalar values id (get scalars id))
                                    (contains? buffers id) (get buffers id)
                                    :else (fail! :emitted-program-host-operand
                                                 "host scalar operand has no runtime binding"
                                                 {:equation (:id equation) :value id}))]))
                       (:operands equation))
                 results (evaluate-host equation
                                        {:operands operand-bindings
                                         :values values})]
             (when-not (map? results)
               (fail! :emitted-program-host-evaluation
                      "host scalar evaluator must return a result map"
                      {:equation (:id equation) :results results}))
             (let [step (validate-host-step!
                         values (->EvaluatedHostEquation equation operand-bindings results))]
               (-> state
                   (assoc :scalars
                          (reduce-kv (fn [environment id value]
                                       (when-let [supplied (get environment id)]
                                         (when-not (semantic-fingerprint/equivalent? supplied value)
                                           (fail! :emitted-program-host-result-conflict
                                                  "supplied and evaluated host scalars disagree"
                                                  {:equation (:id equation) :value id})))
                                       (assoc environment id value))
                                     scalars results))
                   (assoc-in [:host-steps (:id equation)] step)))))))
     {:scalars scalars :device-results #{} :host-steps {}}
     (:equations parallel-program))))

(defn- validate-call-against-program!
  "Check a call against its exact validated program. A constructor may retain exact step objects
   it already checked; public callers pass nil and independently validate every step."
  [call parallel-program validated-equation-calls caller-options]
  (when-not (emitted-program-call? call)
    (fail! :emitted-program-call-type "expected an EmittedParallelProgramCall"
           {:actual (type call)}))
  (when-not (identical? (:program call) parallel-program)
    (fail! :emitted-program-call-program-identity
           "validated program is not the program contained in its call" {}))
  (let [{:keys [steps entry-buffers buffers scalar-values loop-scratch outputs attributes]} call
        current-buffers (volatile! entry-buffers)]
    (doseq [[field value] [[:entry-buffers entry-buffers] [:buffers buffers] [:scalar-values scalar-values]
                           [:loop-scratch loop-scratch] [:outputs outputs]
                           [:attributes attributes]]]
      (when-not (map? value)
        (fail! :emitted-program-call-field "emitted program call fields must be maps"
               {:field field :value value})))
    (when-not (= (count steps) (count (:equations parallel-program)))
      (fail! :emitted-program-call-steps
             "emitted program call must retain one step per equation"
             {:expected (count (:equations parallel-program)) :actual (count steps)}))
    (doseq [[equation step] (map vector (:equations parallel-program) steps)]
      (when (and (seq (:result-views step)) (not (emitted-equation-call? step)))
        (fail! :emitted-program-result-views
               "only emitted numerical equations may declare prefix result views"
               {:equation (:id equation)}))
      (cond
        (evaluated-host-equation? step)
        (do (validate-host-step! (:values parallel-program) step)
            (when-not (= equation (:equation step))
              (fail! :emitted-program-call-step-equation
                     "evaluated host step changed equation identity" {:equation (:id equation)})))

        (emitted-equation-call? step)
        (let [constructed? (and validated-equation-calls
                                (.containsKey ^java.util.IdentityHashMap
                                              validated-equation-calls step))]
          (when-not constructed?
            (if (nil? caller-options)
              (validate-equation-call! step)
              (validate-equation-call! step caller-options)))
            (doseq [[result physical] (:result-views step)]
              (when-not (and (= (get buffers physical) (get (:buffers step) physical))
                             (= (get buffers result) (get (:outputs step) result))
                             (or (not (contains? outputs result))
                                 (= (get buffers result) (get outputs result))))
                (fail! :emitted-program-result-view-bindings
                       "result-view tokens must agree with actual step and exported bindings"
                       {:equation (:id equation) :result result :physical physical})))
            (when-not (= equation (:equation step))
              (fail! :emitted-program-call-step-equation
                     "emitted graph step changed equation identity" {:equation (:id equation)}))
          (if constructed?
            ;; The synchronous constructor retained this exact step's checked post-environment.
            ;; Public and retained-static validators never receive this construction map.
            (vreset! current-buffers
                     (.get ^java.util.IdentityHashMap validated-equation-calls step))
            (let [{expected :bindings next-buffers :buffers}
                  (prepare-equation-bindings equation (:values parallel-program) @current-buffers
                                             scalar-values (:result-views step)
                                             (or *validated-boundary-projections*
                                                 (java.util.IdentityHashMap.)))]
              ;; A certified dispatch may select any admitted graph alternative, but its logical
              ;; runtime arguments must still come from the same source-ordered environment.
              (doseq [field [:buffers :scalar-values :outputs]]
                (let [same? (if (= field :scalar-values)
                              (semantic-fingerprint/equivalent? (get expected field) (get step field))
                              (= (get expected field) (get step field)))]
                  (when-not same?
                    (fail! :emitted-program-call-step-bindings
                           "emitted equation bindings differ from its entry environment"
                           {:equation (:id equation) :field field
                            :expected (get expected field) :actual (get step field)}))))
              (vreset! current-buffers next-buffers))))

        (loop-call/structured-loop-call? step)
        (let [operation (first (:operations equation))
              emitted (if (nil? caller-options)
                        (emitted-loop/validate! operation)
                        (emitted-loop/validate! operation caller-options))]
          (when-not (and (= (:schedule emitted) (:schedule step))
                         (= (:graph emitted) (:graph step)))
            (fail! :emitted-program-call-loop
                   "structured loop call differs from its emitted equation"
                   {:equation (:id equation)}))
          (if (nil? caller-options)
            (loop-call/validate-in-context! step @current-buffers scalar-values loop-scratch)
            (loop-call/validate-in-context! step @current-buffers scalar-values loop-scratch caller-options))
          (vswap! current-buffers merge (:outputs step)))

        :else
        (fail! :emitted-program-call-step "emitted program call has an unknown step"
               {:equation (:id equation) :step step})))
    (when-not (= (set (:outputs parallel-program)) (set (keys outputs)))
      (fail! :emitted-program-call-outputs
             "emitted program call outputs differ from the program boundary"
             {:expected (:outputs parallel-program) :actual (keys outputs)}))
    (when-not (= @current-buffers buffers)
      (fail! :emitted-program-call-final-bindings
             "emitted program final buffers differ from its source-ordered execution"
             {:expected @current-buffers :actual buffers}))
    (doseq [[id actual] outputs]
      (let [expected (if (contains? scalar-values id)
                       (get scalar-values id)
                       (get @current-buffers id ::missing))
            same? (if (typed-scalar? expected)
                    (semantic-fingerprint/equivalent? expected actual)
                    (= expected actual))]
        (when-not same?
          (fail! :emitted-program-call-output-bindings
                 "exported result differs from its source-ordered value"
                 {:value id :expected expected :actual actual}))))
    call))

(defn- validate-for-request!
  "Independently validate a call and its complete emitted program."
  [call caller-options]
  (when-not (emitted-program-call? call)
    (fail! :emitted-program-call-type "expected an EmittedParallelProgramCall"
           {:actual (type call)}))
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        {:keys [program projections]}
        (if (nil? caller-options)
          (emitted-program/validate-with-physical-results! (:program call))
          (emitted-program/validate-with-physical-results! (:program call) caller-options))
        ;; A public validation's synchronous projection scope is mutable and invocation-local;
        ;; the validator's sealed static evidence itself remains read-only.
        local-projections (doto (java.util.IdentityHashMap.) (.putAll projections))
        checked (binding [*validated-boundary-projections* local-projections
                          *validated-projection-policy* policy]
                  (validate-call-against-program! call program nil caller-options))]
    ;; Every public validation starts with a fresh complete program check and projection index.
    ;; A synchronous enclosing rename may retain only facts from this successful check; mapper
    ;; callbacks never inherit that rename context and no index is stored in the returned call.
    (when (and *validated-boundary-projections* (same-projection-request? caller-options))
      (.putAll ^java.util.IdentityHashMap *validated-boundary-projections* local-projections))
    checked))

(defn validate!
  "Independently validate a call and its complete program under external math intent."
  ([call] (validate-for-request! call nil))
  ([call caller-options] (validate-for-request! call caller-options)))

(defn- validate-retained-for-request!
  "Internal exact-owner static-proof path. Recheck every concrete call binding and step;
   reuse only the unchanged emitted program's sealed validation and physical projections.
   The fresh projection scope and evidence are not installed on the returned call."
  [call retained-validation caller-options]
  (when-not (emitted-program-call? call)
    (fail! :emitted-program-call-type "expected an EmittedParallelProgramCall"
           {:actual (type call)}))
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        {:keys [program projections]}
        (if (nil? caller-options)
          (emitted-program/checked-retained-validation! (:program call) retained-validation)
          (emitted-program/checked-retained-validation! (:program call) retained-validation caller-options))
        local-projections (doto (java.util.IdentityHashMap.) (.putAll projections))]
    (binding [*validated-boundary-projections* local-projections
              *validated-projection-policy* policy]
      ;; No constructor exemptions: even steps originating from `make` are checked afresh.
      (validate-call-against-program! call program nil caller-options))))

(defn ^:no-doc validate-with-retained-program!
  "Reuse only exact program evidence checked against this independent request; recheck bindings."
  ([call retained-validation] (validate-retained-for-request! call retained-validation nil))
  ([call retained-validation caller-options]
   (validate-retained-for-request! call retained-validation caller-options)))

(defn- execution-order-for-request
  "Project straight-line selected graph order without allocating device storage.

   Emitted equation calls retain validated, fully selected KernelGraphs. The optional observer
   receives [step-index graph] and supplies the actually bound order instead; both projections
   retain source indices across host equations. Neither projection proves completion or escape.
   Structured control requires a loop-aware witness and deliberately declines."
  [call graph-order caller-options]
   (let [call (if (nil? caller-options) (validate! call) (validate! call caller-options))]
     (when (some loop-call/structured-loop-call? (:steps call))
       (throw (ex-info "structured program execution order requires a loop-aware witness"
                       {:reason :parallel-program-structured-execution-order})))
     (reduce
      (fn [order [step-index step]]
        (if (evaluated-host-equation? step)
          order
          (let [selected (binding [*validated-boundary-projections* nil
                                  *validated-projection-policy* nil]
                           (graph-order step-index (:graph step)))
                annotate #(mapv (fn [entry] (assoc entry :source {:step step-index})) %)]
            (when-not (and (map? selected)
                           (every? #(and (vector? (get selected %))
                                         (every? map? (get selected %)))
                                   [:record-time-prologue :per-replay]))
              (throw (ex-info "graph did not supply a structured execution-order witness"
                              {:reason :parallel-program-graph-order :step step-index})))
            (-> order
                (update :record-time-prologue into (annotate (:record-time-prologue selected)))
                (update :per-replay into (annotate (:per-replay selected)))))))
      {:record-time-prologue [] :per-replay [] :completion :unproven}
      (map-indexed vector (:steps call)))))

(defn- ordinary-graph-order [_ graph]
  {:record-time-prologue []
   :per-replay (mapv #(hash-map :kernel-phase (:id %)) (:nodes graph))})

(defn execution-order
  "Project selected graph order after independent call validation; observers inherit no proof scope."
  ([call] (execution-order-for-request call ordinary-graph-order nil))
  ([call graph-order] (execution-order-for-request call graph-order nil))
  ([call graph-order caller-options] (execution-order-for-request call graph-order caller-options)))

(defn- remap-buffer-map
  [remap buffers]
  (into (empty buffers) (map (fn [[id buffer]] [id (remap buffer)])) buffers))

(defn- remap-loop-call
  [call remap caller-options]
  (let [remap-optional #(when (some? %) (remap %))
        remapped
     (-> call
         (update-in [:buffers :invariants] #(remap-buffer-map remap %))
         (update-in [:buffers :carries]
                    (fn [carries]
                      (mapv #(-> %
                                 (update :initial remap)
                                 (update :output remap)
                                 (update :alternate remap-optional))
                            carries)))
         (update :scratch #(remap-buffer-map remap %))
         (update :outputs #(remap-buffer-map remap %)))]
    (if (nil? caller-options)
      (loop-call/validate! remapped)
      (loop-call/validate! remapped caller-options))))

(defn- buffer-bindings-for-request
  "Return every compiler-value/storage pair referenced by an emitted call boundary.

   A zero-trip structured loop can retain a dead carry-output token that is intentionally absent
   from the call's effective top-level `:buffers`. It is still part of the nested call structure
   and must therefore participate in total storage-identity projections. Graph-owned temporaries
   are not call-boundary storage and are excluded."
  [call caller-options]
  (let [call (if (nil? caller-options) (validate! call) (validate! call caller-options))
        output-bindings (fn [outputs]
                          (remove (comp typed-scalar? second) outputs))
        step-bindings
        (fn [step]
          (cond
            (evaluated-host-equation? step)
            (concat (remove (comp typed-scalar? second) (:operands step))
                    (output-bindings (:outputs step)))

            (emitted-equation-call? step)
            (concat (:buffers step) (output-bindings (:outputs step)))

            (loop-call/structured-loop-call? step)
            (let [invariants (get-in step [:buffers :invariants])
                  carries (get-in step [:buffers :carries])]
              (concat invariants
                      (mapcat (fn [{:keys [initial-id output-id initial output alternate]}]
                                (cond-> [[initial-id initial] [output-id output]]
                                  (some? alternate) (conj [output-id alternate])))
                              carries)
                      (:scratch step)
                      (:outputs step)))

            :else []))]
    (vec (distinct (concat (:entry-buffers call)
                           (:buffers call)
                           (:loop-scratch call)
                           (output-bindings (:outputs call))
                           (mapcat step-bindings (:steps call)))))))

(defn buffer-bindings
  "Return compiler-value/storage pairs after independent request-aware validation."
  ([call] (buffer-bindings-for-request call nil))
  ([call caller-options] (buffer-bindings-for-request call caller-options)))

(defn buffer-identities
  "Return every distinct external/resident storage token referenced by an emitted call."
  ([call] (vec (distinct (map second (buffer-bindings call)))))
  ([call caller-options] (vec (distinct (map second (buffer-bindings call caller-options))))))

(defn- map-buffers-for-request
  "Map every external/resident buffer token in an emitted program call exactly once.

   `f` is a pure storage-identity projection, typically MaterializedBuffer → LinkValue ID. The
   mapping must be total, non-nil, and injective over distinct source storage: this operation may
   rename existing aliases but cannot silently introduce a new alias. Graph-owned temporaries are
   intentionally absent from EmittedParallelProgramCall and remain private to KernelGraph."
  [call f caller-options]
  ;; buffer-identities independently validates the complete input call before projecting it.
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        validated-boundaries (java.util.IdentityHashMap.)
        source-buffers (binding [*validated-boundary-projections* validated-boundaries
                                *validated-projection-policy* policy]
                         (if (nil? caller-options)
                           (buffer-identities call)
                           (buffer-identities call caller-options)))]
    (when-not (ifn? f)
      (fail! :emitted-program-buffer-mapper
             "emitted program buffer remapping requires a callable projection"
             {:mapper f}))
    (let [target-buffers (binding [*validated-boundary-projections* nil
                                 *validated-projection-policy* nil]
                           (mapv f source-buffers))]
      (when-let [source (some (fn [[source target]] (when (nil? target) source))
                              (map vector source-buffers target-buffers))]
        (fail! :emitted-program-buffer-remap-missing
               "emitted program buffer projection returned nil"
               {:source source}))
      (when-not (= (count target-buffers) (count (distinct target-buffers)))
        (fail! :emitted-program-buffer-remap-collision
               "emitted program buffer projection collapsed distinct storage identities"
               {:sources source-buffers :targets target-buffers}))
      (let [mapping (zipmap source-buffers target-buffers)
            remap #(if (contains? mapping %)
                     (get mapping %)
                     (fail! :emitted-program-buffer-remap-untracked
                            "nested emitted call references storage outside the program binding"
                            {:buffer %}))
            remap-host-step
            (fn [step]
              (update step :operands
                      (fn [operands]
                        (into (empty operands)
                              (map (fn [[id value]]
                                     [id (if (typed-scalar? value) value (remap value))]))
                              operands))))
            remap-step
            (fn [step]
              (cond
                (evaluated-host-equation? step) (remap-host-step step)
                (emitted-equation-call? step)
                (-> step
                    (update :buffers #(remap-buffer-map remap %))
                    (update :outputs #(remap-buffer-map remap %)))
                (loop-call/structured-loop-call? step) (remap-loop-call step remap caller-options)))
            remapped
            (-> call
                (update :steps #(mapv remap-step %))
                (update :entry-buffers #(remap-buffer-map remap %))
                (update :buffers #(remap-buffer-map remap %))
                (update :loop-scratch #(remap-buffer-map remap %))
                (update :outputs
                        (fn [outputs]
                          (into (empty outputs)
                                (map (fn [[id value]]
                                       [id (if (typed-scalar? value) value (remap value))]))
                                outputs))))]
        ;; The input was independently validated above and this rewrite retains its exact
        ;; immutable program. Recheck every remapped binding/step against that program without
        ;; deriving its unchanged algorithms and kernels a second time in this construction.
        ;; A later public validation still independently checks the complete program.
        ;; Do not convey the validation context to mapper callbacks (or their futures).
        ;; Its only consumers are the source and final synchronous validation phases.
        (binding [*validated-boundary-projections* validated-boundaries
                  *validated-projection-policy* policy]
          (validate-call-against-program! remapped (:program call) nil caller-options))))))

(defn map-buffers
  "Rename storage under caller intent without granting that request or proof scope to callbacks."
  ([call f] (map-buffers-for-request call f nil))
  ([call f caller-options] (map-buffers-for-request call f caller-options)))

(defn- stage-inputs
  "Fresh checked inputs and host results for this construction only. The evaluator is neither
   returned nor retained. This private value is not an externally reusable validation proof."
  [parallel-program buffers scalar-values loop-scratch evaluate-host result-views retained-validation caller-options]
  (let [{parallel-program :program projections :projections}
        (if retained-validation
          (if (nil? caller-options)
            (emitted-program/checked-retained-validation! parallel-program retained-validation)
            (emitted-program/checked-retained-validation! parallel-program retained-validation caller-options))
          (if (nil? caller-options)
            (emitted-program/validate-with-physical-results! parallel-program)
            (emitted-program/validate-with-physical-results! parallel-program caller-options)))
        _ (when-not (and (map? result-views)
                         (every? (set (mapcat :results
                                             (filter #(emitted-equation/emitted-equation?
                                                       (first (:operations %)))
                                                     (:equations parallel-program))))
                                 (keys result-views)))
            (fail! :emitted-program-result-views
                   "result-view declarations must name emitted numerical results"
                   {:result-views result-views}))
        values (:values parallel-program)
        overlapping-runtime-values (set/intersection (set (keys buffers))
                                                     (set (keys scalar-values)))
        _ (when (seq overlapping-runtime-values)
            (fail! :emitted-program-runtime-kind
                   "one runtime value cannot be both a buffer and a scalar"
                   {:values overlapping-runtime-values}))
        _ (doseq [[id value] scalar-values]
            (when-not (contains? values id)
              (fail! :emitted-program-runtime-value
                     "runtime scalar names an undeclared program value" {:value id}))
            (checked-scalar values id value))
        _ (doseq [[id value] buffers]
            (when-not (contains? values id)
              (fail! :emitted-program-runtime-value
                     "runtime buffer names an undeclared program value" {:value id}))
            (when (nil? value)
              (fail! :emitted-program-buffer "runtime buffer cannot be nil" {:value id})))
        loop-outputs
        (into #{}
              (comp (filter #(emitted-loop/emitted-loop? (first (:operations %))))
                    (mapcat #(map :output (control/carried (:algorithm %)))))
              (:equations parallel-program))
        _ (doseq [[id value] loop-scratch]
            (when-not (contains? loop-outputs id)
              (fail! :emitted-program-loop-scratch
                     "loop scratch must name a structured loop output" {:value id}))
            (when (nil? value)
              (fail! :emitted-program-loop-scratch
                     "loop scratch binding cannot be nil" {:value id})))
        {:keys [scalars host-steps]}
        (binding [*validated-boundary-projections* nil
                  *validated-projection-policy* nil]
          (evaluate-host-equations parallel-program buffers scalar-values evaluate-host))]
    {:program parallel-program :projections projections :buffers buffers :scalars scalars
     :loop-scratch loop-scratch :host-steps host-steps :result-views result-views
     :caller-options caller-options}))

(defn- construct-staged-call
  "Construct only from inputs just staged by this namespace. Host evaluation remains outside
   structural construction; every checked host step is installed unchanged into the call."
  [{parallel-program :program :keys [projections buffers scalars loop-scratch host-steps result-views caller-options]}]
  (let [values (:values parallel-program)
        validated-equation-calls (java.util.IdentityHashMap.)
        planned
        (reduce
         (fn [{:keys [buffers steps]} equation]
           (cond
             (true? (get-in equation [:attributes :host-only]))
             {:buffers buffers :steps (conj steps (get host-steps (:id equation)))}

             (emitted-loop/emitted-loop? (first (:operations equation)))
             (let [operation (first (:operations equation))
                   emitted (if (nil? caller-options)
                             (emitted-loop/validate! operation)
                             (emitted-loop/validate! operation caller-options))
                   call (if (nil? caller-options)
                          (loop-call/make (:schedule emitted) (:graph emitted) buffers scalars loop-scratch)
                          (loop-call/make (:schedule emitted) (:graph emitted)
                                          buffers scalars loop-scratch caller-options))]
               {:buffers (merge buffers (:outputs call))
                :steps (conj steps call)})

             :else
             (let [{:keys [call buffers]}
                   (prepare-equation-call equation values buffers scalars result-views projections)]
               (.put validated-equation-calls call buffers)
               {:buffers buffers :steps (conj steps call)})))
         {:buffers buffers :steps []}
         (:equations parallel-program))
        final-buffers (:buffers planned)
        outputs
        (into {}
              (map (fn [id]
                     [id (cond
                           (contains? scalars id) (get scalars id)
                           (contains? final-buffers id) (get final-buffers id)
                           :else (fail! :emitted-program-output-binding
                                        "program output has no runtime binding" {:value id}))]))
              (:outputs parallel-program))]
    (validate-call-against-program!
     (->EmittedParallelProgramCall
      parallel-program (:steps planned) buffers final-buffers scalars loop-scratch outputs
      {:execution :stage-once-host-repetition :source-inspected false})
     parallel-program validated-equation-calls caller-options)))

(defn make
  "Prepare a source-independent, target-neutral call of an emitted parallel program.

   `buffers` may include preallocated intermediate and output storage in addition to inputs.
   `scalar-values` contains typed runtime scalars. `loop-scratch` maps loop output IDs to alternate
   carry buffers. `evaluate-host` is called only for effect-free scalar equations that do not
   depend on device results.
   Optional `result-views` maps logical results to their prefix-producing physical destinations.
   This declares a storage relation, not proof that arbitrary runtime tokens alias: LinkPlan
   validates concrete views, and runtime preparation requires checked view resolution.
   The seven-argument arity is an internal static-evidence capability path. It skips only repeated
   emitted-program analysis; scalar/buffer/result-view checks and host staging remain fresh.
   The eight-argument arity supplies independent caller options to both fresh and retained paths;
   neither call attributes nor retained evidence may grant math consent."
  ([parallel-program buffers scalar-values loop-scratch evaluate-host]
   (construct-staged-call
    (stage-inputs parallel-program buffers scalar-values loop-scratch evaluate-host {} nil nil)))
  ([parallel-program buffers scalar-values loop-scratch evaluate-host result-views]
   (construct-staged-call
    (stage-inputs parallel-program buffers scalar-values loop-scratch evaluate-host result-views nil nil)))
  ([parallel-program buffers scalar-values loop-scratch evaluate-host result-views retained-validation]
   (construct-staged-call
    (stage-inputs parallel-program buffers scalar-values loop-scratch evaluate-host result-views
                  retained-validation nil)))
  ([parallel-program buffers scalar-values loop-scratch evaluate-host result-views retained-validation caller-options]
   (construct-staged-call
    (stage-inputs parallel-program buffers scalar-values loop-scratch evaluate-host result-views
                  retained-validation caller-options))))
