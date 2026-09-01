(ns raster.compiler.ir.emitted-parallel-program-call
  "Pure runtime binding for one equation-first emitted parallel program.

   Construction evaluates only effect-free host scalar equations that are independent of device
   results. Every numerical equation is then bound from its emitted ABI and retained physical
   result contracts. The resulting call owns no driver handles and never reads retained source."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(defrecord EvaluatedHostEquation [equation operands results])
(defrecord EmittedEquationCall [equation graph buffers scalar-values outputs])
(defrecord EmittedParallelProgramCall
           [program steps buffers scalar-values loop-scratch outputs attributes])

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

(defn- physical-results
  [algorithm]
  (let [facts (soac/facts algorithm)]
    (into {}
          (mapcat (fn [equation]
                    (map vector (nth equation 2) (soac/physical-results facts equation))))
          (soac/equations algorithm))))

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

(defn validate-equation-call!
  [call]
  (when-not (emitted-equation-call? call)
    (fail! :emitted-program-equation-call "expected an EmittedEquationCall" {:call call}))
  (let [{equation :equation call-graph :graph buffers :buffers
         scalar-values :scalar-values outputs :outputs} call
        operation (first (:operations equation))
        emitted (emitted-equation/validate! operation)
        graph (:graph emitted)
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
    (when-not (= (set (:results equation)) (set (keys outputs)))
      (fail! :emitted-program-equation-outputs
             "emitted equation outputs differ from its logical results"
             {:equation (:id equation) :expected (:results equation)
              :actual (keys outputs)}))
    (when-not (= graph call-graph)
      (fail! :emitted-program-equation-graph
             "emitted equation call graph differs from its certified operation"
             {:equation (:id equation)}))
    call))

(defn- prepare-equation-call
  [equation values buffers scalars]
  (let [emitted (emitted-equation/validate! (first (:operations equation)))
        graph (:graph emitted)
        result-storage (physical-results (:algorithm emitted))
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
                        (not= physical-binding logical-binding))
               (fail! :emitted-program-result-alias
                      "logical and physical result bindings disagree"
                      {:equation (:id equation) :result result :physical-result physical}))
             (assoc bindings physical selected result selected)))
         buffers (:results equation))
        runtime-arguments
        (mapv (fn [slot argument]
                (if (= :scalar (:kind slot))
                  (let [value (checked-scalar values argument (get scalars argument))]
                    (when-not (= (dtype/canon (:kernel-dtype slot))
                                 (dtype/canon (:type value)))
                      (fail! :emitted-program-scalar-abi
                             "runtime scalar dtype differs from the emitted graph ABI"
                             {:equation (:id equation) :value argument
                              :expected (:kernel-dtype slot) :actual (:type value)}))
                    value)
                  (require-buffer buffers argument :equation-interface)))
              (:abi graph) (:arguments graph))
        bindings (executable/graph-bindings graph runtime-arguments)
        outputs (select-keys buffers (:results equation))]
    {:call (validate-equation-call!
            (->EmittedEquationCall equation graph (:buffers bindings)
                                   (:scalar-values bindings) outputs))
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
                                         (when-not (= supplied value)
                                           (fail! :emitted-program-host-result-conflict
                                                  "supplied and evaluated host scalars disagree"
                                                  {:equation (:id equation) :value id})))
                                       (assoc environment id value))
                                     scalars results))
                   (assoc-in [:host-steps (:id equation)] step)))))))
     {:scalars scalars :device-results #{} :host-steps {}}
     (:equations parallel-program))))

(defn validate!
  [call]
  (when-not (emitted-program-call? call)
    (fail! :emitted-program-call-type "expected an EmittedParallelProgramCall"
           {:actual (type call)}))
  (let [{:keys [program steps buffers scalar-values loop-scratch outputs attributes]} call
        parallel-program (emitted-program/validate! program)]
    (doseq [[field value] [[:buffers buffers] [:scalar-values scalar-values]
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
      (cond
        (evaluated-host-equation? step)
        (do (validate-host-step! (:values parallel-program) step)
            (when-not (= equation (:equation step))
              (fail! :emitted-program-call-step-equation
                     "evaluated host step changed equation identity" {:equation (:id equation)})))

        (emitted-equation-call? step)
        (do (validate-equation-call! step)
            (when-not (= equation (:equation step))
              (fail! :emitted-program-call-step-equation
                     "emitted graph step changed equation identity" {:equation (:id equation)})))

        (loop-call/structured-loop-call? step)
        (let [step (loop-call/validate! step)
              emitted (emitted-loop/validate! (first (:operations equation)))]
          (when-not (and (= (:schedule emitted) (:schedule step))
                         (= (:graph emitted) (:graph step)))
            (fail! :emitted-program-call-loop
                   "structured loop call differs from its emitted equation"
                   {:equation (:id equation)})))

        :else
        (fail! :emitted-program-call-step "emitted program call has an unknown step"
               {:equation (:id equation) :step step})))
    (when-not (= (set (:outputs parallel-program)) (set (keys outputs)))
      (fail! :emitted-program-call-outputs
             "emitted program call outputs differ from the program boundary"
             {:expected (:outputs parallel-program) :actual (keys outputs)}))
    call))

(defn- remap-buffer-map
  [remap buffers]
  (into (empty buffers) (map (fn [[id buffer]] [id (remap buffer)])) buffers))

(defn- remap-loop-call
  [call remap]
  (let [remap-optional #(when (some? %) (remap %))]
    (loop-call/validate!
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
         (update :outputs #(remap-buffer-map remap %))))))

(defn buffer-bindings
  "Return every compiler-value/storage pair referenced by an emitted call boundary.

   A zero-trip structured loop can retain a dead carry-output token that is intentionally absent
   from the call's effective top-level `:buffers`. It is still part of the nested call structure
   and must therefore participate in total storage-identity projections. Graph-owned temporaries
   are not call-boundary storage and are excluded."
  [call]
  (let [call (validate! call)
        buffer-result? (fn [id] (seq (get-in call [:program :values id :shape])))
        output-bindings (fn [outputs]
                          (keep (fn [[id value]] (when (buffer-result? id) [id value])) outputs))
        step-bindings
        (fn [step]
          (cond
            (evaluated-host-equation? step)
            (concat (keep (fn [[id value]] (when (buffer-result? id) [id value]))
                          (:operands step))
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
    (vec (distinct (concat (:buffers call)
                           (:loop-scratch call)
                           (output-bindings (:outputs call))
                           (mapcat step-bindings (:steps call)))))))

(defn buffer-identities
  "Return every distinct external/resident storage token referenced by an emitted call."
  [call]
  (vec (distinct (map second (buffer-bindings call)))))

(defn map-buffers
  "Map every external/resident buffer token in an emitted program call exactly once.

   `f` is a pure storage-identity projection, typically MaterializedBuffer → LinkValue ID. The
   mapping must be total, non-nil, and injective over distinct source storage: this operation may
   rename existing aliases but cannot silently introduce a new alias. Graph-owned temporaries are
   intentionally absent from EmittedParallelProgramCall and remain private to KernelGraph."
  [call f]
  (let [call (validate! call)]
    (when-not (ifn? f)
      (fail! :emitted-program-buffer-mapper
             "emitted program buffer remapping requires a callable projection"
             {:mapper f}))
    (let [source-buffers (buffer-identities call)
          target-buffers (mapv f source-buffers)]
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
            buffer-result? (fn [id] (seq (get-in call [:program :values id :shape])))
            remap-host-step
            (fn [step]
              (update step :operands
                      (fn [operands]
                        (into (empty operands)
                              (map (fn [[id value]]
                                     [id (if (buffer-result? id) (remap value) value)]))
                              operands))))
            remap-step
            (fn [step]
              (cond
                (evaluated-host-equation? step) (remap-host-step step)
                (emitted-equation-call? step)
                (-> step
                    (update :buffers #(remap-buffer-map remap %))
                    (update :outputs #(remap-buffer-map remap %)))
                (loop-call/structured-loop-call? step) (remap-loop-call step remap)))
            remapped
            (-> call
                (update :steps #(mapv remap-step %))
                (update :buffers #(remap-buffer-map remap %))
                (update :loop-scratch #(remap-buffer-map remap %))
                (update :outputs
                        (fn [outputs]
                          (into (empty outputs)
                                (map (fn [[id value]]
                                       [id (if (buffer-result? id) (remap value) value)]))
                                outputs))))]
        (validate! remapped)))))

(defn make
  "Prepare a source-independent, target-neutral call of an emitted parallel program.

   `buffers` may include preallocated intermediate and output storage in addition to inputs.
   `scalar-values` contains typed runtime scalars. `loop-scratch` maps loop output IDs to alternate
   carry buffers. `evaluate-host` is called only for effect-free scalar equations that do not
   depend on device results."
  [parallel-program buffers scalar-values loop-scratch evaluate-host]
  (let [parallel-program (emitted-program/validate! parallel-program)
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
        (evaluate-host-equations parallel-program buffers scalar-values evaluate-host)
        planned
        (reduce
         (fn [{:keys [buffers steps]} equation]
           (cond
             (true? (get-in equation [:attributes :host-only]))
             {:buffers buffers :steps (conj steps (get host-steps (:id equation)))}

             (emitted-loop/emitted-loop? (first (:operations equation)))
             (let [emitted (emitted-loop/validate! (first (:operations equation)))
                   call (loop-call/make (:schedule emitted) (:graph emitted)
                                        buffers scalars loop-scratch)]
               {:buffers (merge buffers (:outputs call))
                :steps (conj steps call)})

             :else
             (let [{:keys [call buffers]}
                   (prepare-equation-call equation values buffers scalars)]
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
    (validate!
     (->EmittedParallelProgramCall
      parallel-program (:steps planned) final-buffers scalars loop-scratch outputs
      {:execution :stage-once-host-repetition :source-inspected false}))))
