(ns raster.compiler.passes.parallel.structured-control-route
  "Place a typed sequential fixpoint in Raster's common ParallelProgram envelope.

   TypedSOAC and TypedStructuredControl are two algorithm variants, not two program containers.
   This route gives structured control the same ordered equation/value spine used by loop-free
   algorithms. Scheduling replaces the equation's typed control operation with one checked
   ScheduledStructuredLoop; it does not reconstruct or recognize a source-shaped compound loop."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.structured-control-frontend :as frontend]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(defn- ordered-distinct
  [values]
  (vec (distinct values)))

(defn- loop-boundary?
  [equation algorithm]
  (and (control/loop-program? algorithm)
       (= algorithm (control/validate! algorithm))
       (= (:operands equation) (ordered-distinct (control/outer-operands algorithm)))
       (= (:results equation) (control/outer-results algorithm))))

(defn- soac-boundary?
  [equation algorithm]
  (and (soac/program-form? algorithm)
       (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))

(defn- typed-algorithm-boundary?
  [equation algorithm]
  (cond
    (control/loop-program? algorithm)
    (and (loop-boundary? equation algorithm)
         (= [algorithm] (:operations equation)))

    (soac/program-form? algorithm)
    (and (soac-boundary? equation algorithm)
         (or (and (true? (get-in equation [:attributes :host-only]))
                  (empty? (:operations equation)))
             (= (soac/equations algorithm) (:operations equation))))

    :else false))

(defn- scheduled-algorithm-boundary?
  [equation algorithm]
  (cond
    (control/loop-program? algorithm)
    (and (loop-boundary? equation algorithm)
         (= 1 (count (:operations equation)))
         (let [scheduled (first (:operations equation))]
           (and (schedule/scheduled-loop? scheduled)
                (= algorithm (:algorithm (schedule/validate! scheduled))))))

    (soac/program-form? algorithm)
    (and (soac-boundary? equation algorithm)
         (or (and (true? (get-in equation [:attributes :host-only]))
                  (empty? (:operations equation)))
             (and (seq (:operations equation))
                  (every? segop/segop-node? (:operations equation)))))

    :else false))

(defn- typed-operation?
  [operation]
  (or (control/loop-program? operation) (boolean (fusion/equation-info operation))))

(defn- scheduled-operation?
  [operation]
  (or (schedule/scheduled-loop? operation) (segop/segop-node? operation)))

(declare fail!)

(defn validate-typed-program!
  "Validate the exact mixed typed algorithm union admitted by this route."
  [parallel-program]
  (when-not (= :typed-parallel (:dialect parallel-program))
    (fail! :structured-control-program-dialect
           "typed structured-control routing requires :typed-parallel"
           {:dialect (:dialect parallel-program)}))
  (program/validate! parallel-program typed-operation? typed-algorithm-boundary?))

(defn valid-typed-program?
  [parallel-program]
  (try
    (boolean (validate-typed-program! parallel-program))
    (catch clojure.lang.ExceptionInfo _ false)))

(defn validate-scheduled-program!
  "Validate the exact scheduled loop/SegOp union admitted by this route."
  [parallel-program]
  (when-not (= :scheduled-parallel (:dialect parallel-program))
    (fail! :structured-control-program-dialect
           "scheduled structured-control routing requires :scheduled-parallel"
           {:dialect (:dialect parallel-program)}))
  (program/validate! parallel-program scheduled-operation? scheduled-algorithm-boundary?))

(defn valid-scheduled-program?
  [parallel-program]
  (try
    (boolean (validate-scheduled-program! parallel-program))
    (catch clojure.lang.ExceptionInfo _ false)))

(defn- declared-parameter-value
  [program-values array-types scalar-types parameter]
  (let [retained (get program-values parameter)]
    (cond
      (contains? array-types parameter)
      ;; The analyzed algorithm may retain `(unknown-dimension p)` because shape is deliberately
      ;; absent until invocation. The public flat-array boundary has a stronger fact: its one
      ;; dimension is the extent of that exact argument.
      (av/tensor {:dtype (get array-types parameter)
                  :shape [(list 'extent parameter)]
                  :representation (or (:representation retained) {:kind :plain})
                  :memory-space (:memory-space retained)
                  :placement (:placement retained)
                  :sharding (:sharding retained)
                  :ownership (:ownership retained)
                  :effects (or (:effects retained) #{})
                  :attributes (or (:attributes retained) {})})

      (contains? scalar-types parameter)
      (or retained (av/tensor {:dtype (get scalar-types parameter) :shape []}))

      retained retained

      :else
      (fail! :structured-control-public-value
             "public parameter has no retained or declared AbstractValue"
             {:parameter parameter :available (set (keys program-values))}))))

(defn- shape-projection-source
  [expression]
  (loop [expression expression]
    (if (and (seq? expression)
             (descriptor/cast-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
      (recur (first (descriptor/call-args expression)))
      (when (and (seq? expression)
                 (descriptor/alength-op? (descriptor/semantic-op expression))
                 (= 1 (count (descriptor/call-args expression)))
                 (symbol? (first (descriptor/call-args expression))))
        (first (descriptor/call-args expression))))))

(defn- scalar-value?
  [value]
  (and (= :tensor (:kind value)) (empty? (:shape value))))

(defn- dimension-expression
  [dimension]
  (if (and (seq? dimension) (= 'value (first dimension)) (= 2 (count dimension)))
    (second dimension)
    dimension))

(defn- host-computable-dimension?
  [program-values dimension]
  (let [dimension (dimension-expression dimension)
        references (util/free-syms dimension)]
    (or (integer? dimension)
        (and (symbol? dimension) (scalar-value? (get program-values dimension)))
        (and (seq? dimension)
             (= :pure (effects/analyze-effect dimension))
             (every? #(scalar-value? (get program-values %)) references)))))

(defn- certified-shape-dimension
  [program-values source]
  (let [shape (:shape (get program-values source))
        dimension (when (= 1 (count shape)) (first shape))]
    (when (and dimension (host-computable-dimension? program-values dimension))
      (dimension-expression dimension))))

(defn- canonicalize-shape-projection
  [program-values public-inputs symbol expression]
  (let [source (shape-projection-source expression)
        dimension (when (and source (not (contains? public-inputs source)))
                    (certified-shape-dimension program-values source))]
    (if-not dimension
      expression
      (let [result-dtype (:dtype (get program-values symbol))
            dimension-dtype (when (symbol? dimension)
                              (:dtype (get program-values dimension)))]
        (if (or (nil? dimension-dtype) (= result-dtype dimension-dtype))
          dimension
          (list (dtype/scalar-tag-for-dtype result-dtype) dimension))))))

(defn- scalar-equation-expression
  [equation]
  (let [algorithm (:algorithm equation)]
    (when (and (soac/program-form? algorithm)
               (empty? (:effects equation))
               (= 1 (count (:results equation)))
               (= 1 (count (soac/equations algorithm))))
      (let [algorithm-equation (first (soac/equations algorithm))
            {:keys [kind captures lambda]} (soac/operation-parts algorithm-equation)
            {:keys [parameters locals body-results]} (soac/lambda-parts lambda)
            expression (when (and (= 'scalar kind) (empty? locals)
                                  (= 1 (count body-results)))
                         (util/subst-syms (zipmap parameters captures)
                                          (first body-results)))]
        expression))))

(defn- host-invocation-equation?
  [available program-outputs equation]
  (let [expression (scalar-equation-expression equation)]
    (and expression
         (not-any? (set (:results equation)) program-outputs)
         (set/subset? (set (:operands equation)) available)
         (or (shape-projection-source expression)
             (= :pure (effects/analyze-effect expression))))))

(defn- hoist-host-invocation-equations
  [parallel-program]
  (let [program-outputs (:outputs parallel-program)
        {:keys [hoisted retained]}
        (reduce (fn [{:keys [available] :as state} equation]
                  (let [host? (host-invocation-equation? available program-outputs equation)
                        graph-shape? (true? (get-in equation
                                                    [:attributes :graph-shape-definition]))]
                    (cond
                      (and host? (not graph-shape?))
                      (-> state
                          (update :hoisted conj equation)
                          (update :available into (:results equation)))

                      (and host? graph-shape?)
                      ;; Allocation consumes this scalar on the host and target narrowing
                      ;; consumes its exact typed equation as proof. Keep one explicit host-only
                      ;; equation in program order; the invocation prefix materializes the same
                      ;; pure definition before allocating storage.
                      (-> state
                          (update :retained conj
                                  (-> equation
                                      (assoc :operations [])
                                      (update :attributes assoc :host-only true)))
                          (update :available into (:results equation)))

                      :else
                      (update state :retained conj equation))))
                {:available (set (:inputs parallel-program))
                 :hoisted [] :retained []}
                (:equations parallel-program))]
    (if (and (empty? hoisted) (= retained (:equations parallel-program)))
      parallel-program
      (let [inputs (program/infer-inputs retained)
            shape-equations (count (filter (comp shape-projection-source
                                                 scalar-equation-expression)
                                           hoisted))]
        (-> parallel-program
            (assoc :equations retained
                   :inputs inputs
                   :effects (reduce set/union #{} (map :effects retained)))
            (update :attributes assoc
                    :invocation-scalar-equations (count hoisted)
                    :invocation-shape-equations shape-equations))))))

(defn- required-prefix-symbols
  [pairs roots]
  (let [definitions (into {} (map (fn [[symbol expression]] [symbol expression])) pairs)
        available (set (keys definitions))]
    (loop [required (set/intersection available (set roots))]
      (let [dependencies (->> required
                              (mapcat #(util/free-syms (get definitions %)))
                              set
                              (set/intersection available))
            required' (set/union required dependencies)]
        (if (= required required') required (recur required'))))))

(defn- invocation-prefix
  [parallel-program public-parameters prefix-values]
  (let [source (:source parallel-program)]
    (when-not (and (seq? source) (contains? #{'let 'let*} (first source)))
      (fail! :structured-control-invocation-source
             "an analyzed TypedSOAC program requires a flat retained host source boundary"
             {:source source}))
    (let [program-values (:values parallel-program)
          prefix-values (merge prefix-values program-values)
          public-inputs (set public-parameters)
          source-pairs (mapv (fn [[symbol expression]]
                               [symbol (canonicalize-shape-projection
                                        prefix-values public-inputs symbol expression)])
                             (partition 2 (second source)))
          equation-sites (into #{} (keep (fn [equation]
                                           (when (= :binding (first (:site equation)))
                                             (second (:site equation)))))
                               (:equations parallel-program))
          graph-shape-definitions
          (into #{} (comp (filter #(true? (get-in %
                                                   [:attributes :graph-shape-definition])))
                          (mapcat :results))
                (:equations parallel-program))
          candidates (filterv (fn [[symbol _]]
                                ;; A graph-shape scalar remains an ordered TypedSOAC equation so
                                ;; target narrowing can prove its exact algebra.  Host allocation
                                ;; also needs the same pure value before numerical execution, so
                                ;; materialize that one definition in the invocation prefix too.
                                (or (contains? graph-shape-definitions symbol)
                                    (not (contains? equation-sites symbol))))
                              source-pairs)
          shape-roots
          (into #{}
                (comp (mapcat (comp #(mapcat util/free-syms %) :shape val))
                      (filter #(scalar-value? (get prefix-values %))))
                program-values)
          roots (set/difference (set/union (set (:inputs parallel-program)) shape-roots)
                                (set public-parameters))
          required (required-prefix-symbols candidates roots)
          bindings (filterv (comp required first) candidates)
          produced (set (map first bindings))
          missing (set/difference roots produced)]
      (when (seq missing)
        (fail! :structured-control-invocation-prefix
               "internal TypedSOAC inputs are not public parameters or host-prefix values"
               {:inputs missing :public-parameters (set public-parameters)
                :available produced}))
      bindings)))

(defn- normalize-public-contracts
  [parallel-program parameter-values]
  (let [normalize-values (fn [values]
                           (reduce-kv (fn [values id value]
                                        (if (and (contains? values id)
                                                 (some (fn [dimension]
                                                         (and (seq? dimension)
                                                              (= 'unknown-dimension
                                                                 (first dimension))))
                                                       (:shape (get values id))))
                                          (assoc values id value)
                                          values))
                                      values parameter-values))]
    (update parallel-program :values normalize-values)))

(defn promote-soac-program
  "Promote one analyzed, loop-free TypedSOAC ParallelProgram into the common typed program union.

   The numerical equations are unchanged. This pass adds only the public invocation contract:
   direct parameters map by identity, while the transitive non-equation host prefix becomes typed
   ShapeProjection/ScalarCompute/allocation steps. Structured loops and loop-free algorithms can
   therefore share scheduling, emission, ProgramCall, LinkPlan, and runtime lowering."
  [parallel-program {:keys [public-parameters active-params array-types scalar-types]
                     :as options}]
  (when-not (= :typed-soac (:dialect parallel-program))
    (fail! :structured-control-soac-promotion
           "loop-free promotion requires an analyzed :typed-soac ParallelProgram"
           {:dialect (:dialect parallel-program)}))
  (let [parallel-program (hoist-host-invocation-equations parallel-program)
        public-parameters (vec (or public-parameters active-params))
        _ (when-not (seq public-parameters)
            (fail! :structured-control-public-parameters
                   "loop-free equation-first promotion requires ordered public parameters" {}))
        retained-values (:values parallel-program)
        parameter-values (into {} (map (fn [parameter]
                                         [parameter
                                          (declared-parameter-value
                                           retained-values array-types scalar-types parameter)]))
                               public-parameters)
        parallel-program (normalize-public-contracts parallel-program parameter-values)
        program-values (:values parallel-program)
        host-values
        (:values
         (host-av/analyze
          (:source parallel-program)
          (assoc options :values program-values
                 :array-types array-types :scalar-types scalar-types)))
        ;; Host relational analysis is authoritative only where it genuinely refines the typed
        ;; program (not where it merely chooses a different symbolic spelling). This supplies flat
        ;; physical allocation extents while preserving logical N-D result shapes.
        program-values
        (reduce-kv (fn [values symbol host-value]
                     (if-let [program-value (get values symbol)]
                       (if-let [refined (av/merge-refinement program-value host-value)]
                         (assoc values symbol refined)
                         values)
                       values))
                   program-values host-values)
        ;; Symbols that occur only as logical/physical dimensions are still typed values at the
        ;; public invocation boundary. Declare their retained host facts in the program value
        ;; table so emitted calls can certify N-D logical shapes against flattened storage without
        ;; inventing an untyped shape environment in the linker.
        shape-symbols (into #{} (mapcat (comp #(mapcat util/free-syms %) :shape val))
                            program-values)
        shape-values (into {} (filter (fn [[symbol value]]
                                        (and (contains? shape-symbols symbol)
                                             (scalar-value? value))))
                           host-values)
        program-values (merge shape-values program-values)
        parallel-program (assoc parallel-program :values program-values)
        prefix (invocation-prefix parallel-program public-parameters host-values)
        binding-values
        (into {}
              (map (fn [[symbol expression]]
                     (let [program-value (get program-values symbol)
                           host-value (get host-values symbol)
                           value (cond
                                   ;; An exact lexical alias keeps one runtime storage identity.
                                   ;; Its program-side symbolic shape remains authoritative and
                                   ;; materialization later proves both symbolic extents resolve to
                                   ;; the same concrete view before binding.
                                   (and program-value host-value (symbol? expression))
                                   program-value

                                   (and program-value host-value)
                                   (or (av/merge-refinement program-value host-value)
                                       (when (av/storage-contract-compatible?
                                              program-value host-value)
                                         host-value)
                                       (fail! :structured-control-prefix-value-conflict
                                              "host and typed program facts disagree on a prefix value"
                                              {:symbol symbol :expression expression
                                               :program-value program-value
                                               :host-value host-value}))
                                   program-value program-value
                                   host-value host-value)]
                       (when-not value
                         (fail! :structured-control-prefix-value
                                "host-prefix binding has no retained AbstractValue"
                                {:symbol symbol :expression expression}))
                       [symbol value])))
              prefix)
        program-values
        (reduce-kv (fn [values symbol value]
                     (if-let [prior (get values symbol)]
                       (if-let [refined (av/merge-refinement prior value)]
                         (assoc values symbol refined)
                         values)
                       values))
                   program-values binding-values)
        parallel-program (assoc parallel-program :values program-values)
        promoted (assoc parallel-program :dialect :typed-parallel)
        public-set (set public-parameters)
        external-result-storage
        (->> (:equations parallel-program)
             (mapcat #(get-in % [:attributes :result-storage]))
             (map :destination)
             (filter public-set)
             (remove (set (:inputs parallel-program)))
             distinct
             vec)
        plan (invocation/from-prefix
              {:id [:program-invocation (mapv :id (:equations parallel-program))]
               :parameters public-parameters
               :parameter-values parameter-values
               :bindings prefix
               :binding-values binding-values
               :program-values program-values
               :program-inputs (:inputs parallel-program)
               :program-storage external-result-storage
               :program-outputs (:outputs parallel-program)
               :attributes {:source-dialect :closed-clojure
                            :algorithm-dialect :typed-soac
                            :target-dialect :typed-invocation}})]
    (-> promoted
        (update :attributes assoc
                :host-control :typed-invocation
                :invocation-plan (invocation/validate-against! plan promoted))
        validate-typed-program!)))

(defn- merge-values
  [left right]
  (reduce-kv
   (fn [values id value]
     (if-let [prior (get values id)]
       (if-let [refined (av/merge-refinement prior value)]
         (assoc values id refined)
         (throw (ex-info "mixed typed algorithms disagree on one AbstractValue"
                         {:reason :structured-control-value-conflict
                          :id id :first prior :second value})))
       (assoc values id value)))
   left right))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :structured-control-route))))

(defn- program-boundary
  [equations outputs]
  {:inputs (program/infer-inputs equations)
   :outputs (vec outputs)})

(defn- value-types
  [values shape?]
  (into {} (keep (fn [[id value]]
                   (when (shape? (:shape value)) [id (:dtype value)]))) values))

(defn- suffix-program
  [{:keys [loop suffix-bindings outer-body]} {:keys [dtype abstract-machine] :as options}]
  (when (or (seq suffix-bindings) (seq outer-body))
    (let [substitutions (into {} (map (juxt :initial :output) (control/carried loop)))
          source (util/subst-syms
                  substitutions
                  (list* 'let* (vec (mapcat identity suffix-bindings)) outer-body))
          values (control/outer-values loop)
          typed (typed-frontend/form->program
                 source (assoc options
                               :dtype (or dtype :double)
                               :values values
                               :array-types (merge (:array-types options)
                                                   (value-types values seq))
                               :scalar-types (merge (:scalar-types options)
                                                    (value-types values empty?))))]
      (when-not typed
        (fail! :structured-control-unsupported-suffix
               "post-loop computation is outside the certified TypedSOAC subset"
               {:suffix-bindings suffix-bindings :outer-body outer-body}))
      (first (fusion/fusion-fixpoint typed abstract-machine)))))

(defn- loop-equation
  [loop loop-binding]
  (let [facts (control/facts loop)]
    (program/->ProgramEquation
     [:structured-loop (:id facts)] [:binding loop-binding] nil
     (ordered-distinct (control/outer-operands loop))
     (control/outer-results loop) loop [loop]
     (:effects facts)
     (assoc (:provenance facts) :pass :structured-control-route)
     (assoc (:attributes facts) :algorithm-dialect :typed-structured-control))))

(defn program-envelope
  "Wrap one certified frontend decomposition in the shared typed program envelope.

   When a suffix is itself in the TypedSOAC subset, its uses of the physical carried array are
   alpha-remapped to the loop's fresh logical output and appended as ordinary typed equations."
  ([decomposition] (program-envelope decomposition {}))
  ([{:keys [loop source loop-binding prefix-bindings parameter-values outer-values]
     :as decomposition} options]
   (when-not (and (map? decomposition) loop)
     (throw (ex-info "structured-control routing requires a certified frontend decomposition"
                     {:reason :structured-control-decomposition
                      :decomposition decomposition})))
   (let [loop (control/validate! loop)
         loop-equation (loop-equation loop loop-binding)
         suffix (suffix-program decomposition options)
         suffix-envelope (when suffix (typed-route/program-envelope suffix))
         suffix-equations (mapv #(update % :id (fn [id] [:suffix id]))
                                (:equations suffix-envelope))
         equations (into [loop-equation] suffix-equations)
         values (merge-values (control/outer-values loop) (:values suffix-envelope))
         program-outputs (or (:outputs suffix-envelope) (control/outer-results loop))
         {:keys [inputs outputs]} (program-boundary equations program-outputs)
         effects (reduce set/union #{} (map :effects equations))
         parallel-program
         (program/make
          {:dialect :typed-parallel
           :source source
           :values values
           :inputs inputs
           :equations equations
           :outputs outputs
           :effects effects
           :provenance {:source-dialect :typed-structured-control
                        :pass :structured-control-route}
           :attributes {:host-control :typed-structured-control
                        :mixed-algorithms (boolean suffix)}
           :operation? typed-operation?
           :algorithm? typed-algorithm-boundary?})
         public-parameters (or (:public-parameters options) (:active-params options))]
     (if (seq public-parameters)
       (let [plan (invocation/from-prefix
                   {:id [:program-invocation (:id (control/facts loop))]
                    :parameters public-parameters
                    :parameter-values parameter-values
                    :bindings prefix-bindings
                    :binding-values outer-values
                    :program-values values
                    :program-inputs inputs
                    :program-outputs outputs
                    :attributes {:source-dialect :closed-clojure
                                 :target-dialect :typed-invocation}})]
         (-> parallel-program
             (assoc-in [:attributes :invocation-plan]
                       (invocation/validate-against! plan parallel-program))
             validate-typed-program!))
       parallel-program))))

(defn attempt
  "Attempt the complete structured-control semantic route.

   Absence means the source is outside the generic counted-loop shape. An unsupported post-loop
   continuation is an explicit coverage decline and leaves the caller free to preserve the
   original source; contradictions after certification still escape."
  [source options]
  (when-let [decomposition (frontend/form->structured-loop source options)]
    (try
      (let [parallel-program (program-envelope decomposition options)
            equations (:equations parallel-program)]
        {:program parallel-program
         :stats {:route :typed-structured-control
                 :typed-validated true
                 :structured-loop-equations
                 (count (filter #(control/loop-program? (:algorithm %)) equations))
                 :typed-soac-equations
                 (count (filter #(soac/program-form? (:algorithm %)) equations))}})
      (catch clojure.lang.ExceptionInfo exception
        (if (= :structured-control-unsupported-suffix (:reason (ex-data exception)))
          {:declined (select-keys (ex-data exception) [:reason :suffix-bindings :outer-body])}
          (throw exception))))))

(defn schedule-program
  "Schedule every typed-control equation through the ordinary loop-body SOAC vertical."
  [parallel-program opts]
  (let [parallel-program
        (validate-typed-program! parallel-program)
        {:keys [equations values]}
        (reduce
         (fn [{:keys [equations values]} equation]
           (let [algorithm (:algorithm equation)]
             (cond
               (true? (get-in equation [:attributes :host-only]))
               {:equations (conj equations equation) :values values}

               (control/loop-program? algorithm)
               (let [scheduled (lower/schedule algorithm opts)]
                 {:equations
                  (conj equations
                        (-> equation
                            (assoc :operations [scheduled])
                            (update :provenance assoc
                                    :target-dialect :structured-control-schedule)
                            (update :attributes assoc
                                    :schedule-dialect :segop
                                    :graph-dialect :kernel-graph)))
                  :values values})

               :else
               (let [algorithm-values (:values (soac/facts algorithm))
                     schedule-options
                     (assoc opts
                            :array-types (merge (:array-types opts)
                                                (value-types algorithm-values seq))
                            :scalar-types (merge (:scalar-types opts)
                                                 (value-types algorithm-values empty?)))
                     scheduled (:form (segop-lower/segop-lower-pass
                                       (typed-route/program-envelope algorithm)
                                       schedule-options))
                     _ (when-not (= 1 (count (:equations scheduled)))
                         (fail! :structured-control-suffix-schedule
                                "one mixed suffix equation must lower to one scheduled equation"
                                {:equation (:id equation)
                                 :scheduled-equations (mapv :id (:equations scheduled))}))
                     scheduled-equation (first (:equations scheduled))]
                 {:equations
                  (conj equations
                        (-> equation
                            (assoc :operations (:operations scheduled-equation))
                            (update :provenance merge (:provenance scheduled-equation))
                            (update :attributes merge (:attributes scheduled-equation))))
                  :values (merge-values values (:values scheduled))}))))
         {:equations [] :values (:values parallel-program)}
         (:equations parallel-program))]
    (validate-scheduled-program!
     (program/make
      {:dialect :scheduled-parallel
       :source (:source parallel-program)
       :values values
       :inputs (:inputs parallel-program)
       :equations equations
       :outputs (:outputs parallel-program)
       :effects (:effects parallel-program)
       :diagnostics (:diagnostics parallel-program)
       :provenance (assoc (:provenance parallel-program)
                          :target-dialect :scheduled-parallel)
       :attributes (:attributes parallel-program)
       :operation? scheduled-operation?
       :algorithm? scheduled-algorithm-boundary?}))))
