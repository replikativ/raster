(ns raster.compiler.ir.parallel-program
  "First-class typed program container for parallel compiler dialects.

   ParallelProgram is the ordered value/equation spine shared by the SOAC and SegOp stages.  The
   current compiler still needs `source` to reconstruct scalar host control around emitted kernel
   calls, but scheduled operations live in equations, never in Clojure symbol metadata.  As more
   scalar control becomes explicit, `source` can shrink without changing value or operation IDs."
  (:require [clojure.set :as set]
            [raster.compiler.ir.abstract-value :as av]))

(defrecord ProgramEquation
           [id site source operands results algorithm operations effects provenance attributes])

(defrecord ParallelProgram
           [dialect source values inputs equations outputs effects diagnostics provenance attributes])

(defn- record-kind?
  [record-class value]
  (and value (= record-class (.getName (class value)))))

(defn equation?
  [value]
  (record-kind? "raster.compiler.ir.parallel_program.ProgramEquation" value))

(defn parallel-program?
  [value]
  (record-kind? "raster.compiler.ir.parallel_program.ParallelProgram" value))

(defn- distinct-vector?
  [value]
  (and (vector? value) (= (count value) (count (distinct value)))))

(defn infer-inputs
  "Infer external inputs in first-use order from an ordered equation sequence.

   This is a constructor helper, not a substitute for validation. A value used before a later
   definition is inferred as external and `validate!` subsequently rejects that definition as an
   input clobber."
  [equations]
  (:inputs
   (reduce (fn [{:keys [inputs available] :as state} equation]
             (let [external (remove available (:operands equation))]
               (-> state
                   (update :inputs into (remove (set inputs) external))
                   (update :available into external)
                   (update :available into (:results equation)))))
           {:inputs [] :available #{}}
           equations)))

(defn- validate-dataflow!
  [{:keys [inputs equations outputs] :as program}]
  (let [available
        (reduce
         (fn [available equation]
           (let [missing (set/difference (set (:operands equation)) available)
                 redefined (set/intersection (set (:results equation)) available)]
             (when (seq missing)
               (throw (ex-info "equation operands must be program inputs or earlier results"
                               {:reason :parallel-program-use-before-definition
                                :equation (:id equation) :values missing})))
             (when (seq redefined)
               (throw (ex-info "equation results must be fresh logical values"
                               {:reason :parallel-program-result-redefinition
                                :equation (:id equation) :values redefined})))
             (into available (:results equation))))
         (set inputs) equations)
        unavailable (set/difference (set outputs) available)]
    (when (seq unavailable)
      (throw (ex-info "program outputs must be inputs or equation results"
                      {:reason :parallel-program-unavailable-output
                       :values unavailable})))
    program))

(defn validate!
  "Validate a ParallelProgram and return it unchanged.

   `operation?`, when supplied, defines full legality for the declared scheduled dialect.
   `algorithm?` receives `[equation algorithm]`, so a caller can validate an optional semantic
   dialect and its boundary. Structural validation remains independent of concrete SOAC/SegOp
   namespaces, avoiding an IR cycle."
  ([program] (validate! program (constantly true) (fn [_ _] true)))
  ([program operation?] (validate! program operation? (fn [_ _] true)))
  ([program operation? algorithm?]
   (when-not (parallel-program? program)
     (throw (ex-info "expected a ParallelProgram"
                     {:reason :parallel-program-type :actual (type program)})))
   (let [{:keys [dialect values inputs equations outputs effects diagnostics provenance attributes]}
         program]
     (when-not (keyword? dialect)
       (throw (ex-info "parallel program requires a dialect keyword"
                       {:reason :parallel-program-dialect :dialect dialect})))
     (when-not (map? values)
       (throw (ex-info "parallel program values must be an id-to-AbstractValue map"
                       {:reason :parallel-program-values :values values})))
     (doseq [[id value] values]
       (when (nil? id)
         (throw (ex-info "parallel program value IDs cannot be nil"
                         {:reason :parallel-program-value-id})))
       (av/validate! value))
     (doseq [[field ids] [[:inputs inputs] [:outputs outputs]]]
       (when-not (distinct-vector? ids)
         (throw (ex-info "parallel program boundary IDs must be a distinct vector"
                         {:reason :parallel-program-boundary :field field :ids ids})))
       (doseq [id ids]
         (when-not (contains? values id)
           (throw (ex-info "parallel program boundary references an unknown value"
                           {:reason :parallel-program-unknown-value :field field :id id})))))
     (when-not (vector? equations)
       (throw (ex-info "parallel program equations must be an ordered vector"
                       {:reason :parallel-program-equations :equations equations})))
     (when-not (= (count equations) (count (distinct (map :id equations))))
       (throw (ex-info "parallel program equation IDs must be unique"
                       {:reason :parallel-program-equation-id})))
     (doseq [equation equations]
       (when-not (equation? equation)
         (throw (ex-info "parallel program contains a non-equation"
                         {:reason :parallel-program-equation-type :equation equation})))
       (doseq [[field ids] [[:operands (:operands equation)] [:results (:results equation)]]]
         (when-not (distinct-vector? ids)
           (throw (ex-info "equation value IDs must be a distinct vector"
                           {:reason :parallel-program-equation-values
                            :equation (:id equation) :field field :ids ids})))
         (doseq [id ids]
           (when-not (contains? values id)
             (throw (ex-info "equation references an unknown value"
                             {:reason :parallel-program-unknown-value
                              :equation (:id equation) :field field :id id})))))
       (when-not (and (vector? (:operations equation))
                      (or (seq (:operations equation))
                          (true? (get-in equation [:attributes :host-only]))))
         (throw (ex-info "equation requires ordered operations or an explicit host-only contract"
                         {:reason :parallel-program-operations :equation (:id equation)})))
       (when (and (true? (get-in equation [:attributes :host-only]))
                  (seq (:operations equation)))
         (throw (ex-info "host-only equations cannot carry scheduled operations"
                         {:reason :parallel-program-host-only-operations
                          :equation (:id equation)})))
       (doseq [operation (:operations equation)]
         (when-not (operation? operation)
           (throw (ex-info "an illegal operation remains after full dialect conversion"
                           {:reason :illegal-op-remains :target-dialect dialect
                            :equation (:id equation) :operation operation :fallback :none}))))
       (when (and (some? (:algorithm equation))
                  (not (algorithm? equation (:algorithm equation))))
         (throw (ex-info "equation contains an illegal algorithm value"
                         {:reason :parallel-program-algorithm
                          :target-dialect dialect
                          :equation (:id equation)
                          :algorithm (:algorithm equation)})))
       (when-not (set? (:effects equation))
         (throw (ex-info "equation effects must be an explicit set"
                         {:reason :parallel-program-effects :equation (:id equation)})))
       (when-not (map? (:provenance equation))
         (throw (ex-info "equation provenance must be a map"
                         {:reason :parallel-program-provenance :equation (:id equation)})))
       (when-not (map? (:attributes equation))
         (throw (ex-info "equation attributes must be a map"
                         {:reason :parallel-program-attributes :equation (:id equation)}))))
     (validate-dataflow! program)
     (when-not (set? effects)
       (throw (ex-info "parallel program effects must be an explicit set"
                       {:reason :parallel-program-effects :effects effects})))
     (when-not (vector? diagnostics)
       (throw (ex-info "parallel program diagnostics must be a vector"
                       {:reason :parallel-program-diagnostics :diagnostics diagnostics})))
     (when-not (map? provenance)
       (throw (ex-info "parallel program provenance must be a map"
                       {:reason :parallel-program-provenance :provenance provenance})))
     (when-not (map? attributes)
       (throw (ex-info "parallel program attributes must be a map"
                       {:reason :parallel-program-attributes :attributes attributes}))))
   program))

(defn make
  [{:keys [dialect source values inputs equations outputs effects diagnostics provenance attributes
           operation? algorithm?]
    :or {values {} inputs [] equations [] outputs [] effects #{} diagnostics []
         provenance {} attributes {} operation? (constantly true) algorithm? (fn [_ _] true)}}]
  (validate! (->ParallelProgram dialect source values (vec inputs) (vec equations) (vec outputs)
                                effects (vec diagnostics) provenance attributes)
             operation? algorithm?))

(defn source-form
  "The compatibility host expression surrounding the explicit parallel equations."
  [program]
  (:source (validate! program)))

(defn declared-value-types
  "Project declared dtypes for an explicit collection of value IDs.

   The caller supplies ABI/storage roles from its scheduled operation. Logical rank is deliberately
   not used to guess whether a value is passed by value or by buffer: a rank-zero tensor may later
   be realized either way."
  [program ids]
  (let [values (:values (validate! program))]
    (into {}
          (map (fn [id]
                 (let [value (get values id)]
                   (when-not (and (symbol? id) value (keyword? (:dtype value)))
                     (throw (ex-info "scheduled parameter lacks a declared symbolic value dtype"
                                     {:reason :parallel-program-parameter-dtype
                                      :id id :value value})))
                   [id (:dtype value)])))
          ids)))

(defn equation-for-binding
  "Find the equation for a binding only when the current source still matches its certified source.
   This equality guard invalidates scheduled operations if a later backend-local fusion rewrites the
   binding before consumption."
  [program sym source]
  (some #(when (and (= [:binding sym] (:site %)) (= source (:source %))) %) (:equations program)))

(defn equation-for-source
  "Find an unambiguous body-position equation for an exact source expression."
  [program source]
  (let [matches (filterv #(and (= :body (first (:site %))) (= source (:source %)))
                         (:equations program))]
    (when (= 1 (count matches)) (first matches))))

(defn operations-for-binding
  [program sym source]
  (:operations (equation-for-binding program sym source)))

(defn algorithm-for-binding
  [program sym source]
  (:algorithm (equation-for-binding program sym source)))

(defn result-id-for-binding
  "The explicit value ID produced by a binding equation, when it has one result."
  [program sym source]
  (first (:results (equation-for-binding program sym source))))

(defn operations-for-source
  [program source]
  (:operations (equation-for-source program source)))

(defn algorithm-for-source
  [program source]
  (:algorithm (equation-for-source program source)))

(defn kernel-graph-for-binding
  [program sym source]
  (:kernel-graph (:attributes (equation-for-binding program sym source))))
