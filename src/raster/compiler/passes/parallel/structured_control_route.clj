(ns raster.compiler.passes.parallel.structured-control-route
  "Place a typed sequential fixpoint in Raster's common ParallelProgram envelope.

   TypedSOAC, TypedStructuredControl, and the existing generic segmented weighted-reduction plan
   are algorithm variants, not separate program containers. This route gives them the same ordered
   equation/value spine used by loop-free algorithms. Scheduling replaces a typed operation with a
   checked schedule; it does not reconstruct or recognize source-shaped compound operations."
  (:require [clojure.set :as set]
            [raster.compiler.core.callee-identity :as callee-identity]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.device-scalar-schedule :as device-scalar]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.indexed-weighted-reduction-body :as indexed-body]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
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
  [values equation algorithm]
  (cond
    (control/loop-program? algorithm)
    (and (loop-boundary? equation algorithm)
         (= [algorithm] (:operations equation)))

    (soac/program-form? algorithm)
    (and (soac-boundary? equation algorithm)
         (or (and (true? (get-in equation [:attributes :host-only]))
                  (empty? (:operations equation)))
             (= (soac/equations algorithm) (:operations equation))))

    (swr/plan? algorithm)
    (and (= [algorithm] (:operations equation))
         (boolean (swr/equation-boundary? values equation algorithm)))

    :else false))

(defn- scheduled-algorithm-boundary?
  [parallel-program equation algorithm]
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

    (swr/plan? algorithm)
    (and (swr/equation-boundary? (:values parallel-program) equation algorithm)
         (= 1 (count (:operations equation)))
         (let [certificate (first (:operations equation))]
           (and (scheduled-body/scheduled-kernel-body? certificate)
                (let [graph (:graph (equation-graph/make-for-plan-equation
                                     parallel-program equation))]
                  (boolean (scheduled-body/validate-against-node!
                            certificate (first (:nodes graph)) graph))))))

    :else false))

(defn- typed-operation?
  [operation]
  (or (control/loop-program? operation)
      (swr/plan? operation)
      (boolean (fusion/equation-info operation))))

(defn- scheduled-operation?
  [operation]
  (or (schedule/scheduled-loop? operation) (segop/segop-node? operation)
      (scheduled-body/scheduled-kernel-body? operation)))

(declare fail!)

(defn validate-typed-program!
  "Validate the exact mixed typed algorithm union admitted by this route."
  [parallel-program]
  (when-not (= :typed-parallel (:dialect parallel-program))
    (fail! :structured-control-program-dialect
           "typed structured-control routing requires :typed-parallel"
           {:dialect (:dialect parallel-program)}))
  (program/validate! parallel-program typed-operation?
                     (partial typed-algorithm-boundary? (:values parallel-program))))

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
  (program/validate! parallel-program scheduled-operation?
                     (partial scheduled-algorithm-boundary? parallel-program)))

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
             (not= :wrap (descriptor/cast-integral-narrowing (descriptor/semantic-op expression)))
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
  [available program-outputs required-shape-scalars source-prefix? equation]
  (let [expression (scalar-equation-expression equation)]
    (and expression
         (not-any? (set (:results equation)) program-outputs)
         (set/subset? (set (:operands equation)) available)
         ;; Invocation-prefix construction is dependency-pruned and executes before all device
         ;; equations. Even a shape projection may not cross that boundary when a checked cast
         ;; still carries exceptional control. A checked scalar which really precedes device work
         ;; may cross only when a retained shape requires its value before launch; the attested
         ;; InvocationPlan region then becomes the exceptional computation's single owner.
         (or (effects/removable-expr? expression)
             (and source-prefix?
                  (true? (get-in equation [:attributes :source-prefix?]))
                  (some required-shape-scalars (:results equation)))))))

(defn- required-shape-scalars
  "Return scalar equation results in the transitive definition closure of retained shapes."
  [parallel-program]
  (let [shape-symbols
        (into #{}
              (mapcat (fn [value]
                        (mapcat util/free-syms (:shape value))))
              (vals (:values parallel-program)))
        definitions
        (into {}
              (mapcat (fn [equation]
                        (when (scalar-equation-expression equation)
                          (map (fn [result] [result equation]) (:results equation)))))
              (:equations parallel-program))]
    (loop [required (set/intersection shape-symbols (set (keys definitions)))]
      (let [dependencies
            (into required
                  (comp (mapcat #(get-in definitions [% :operands]))
                        (filter #(contains? definitions %)))
                  required)]
        (if (= required dependencies) required (recur dependencies))))))

(defn- hoist-host-invocation-equations
  [parallel-program]
  (let [program-outputs (:outputs parallel-program)
        shape-scalars (required-shape-scalars parallel-program)
        {:keys [hoisted retained]}
        (reduce (fn [{:keys [available source-prefix?] :as state} equation]
                  (let [expression (scalar-equation-expression equation)
                        scalar? (some? expression)
                        required-shape? (some shape-scalars (:results equation))
                        exceptional-shape? (and required-shape?
                                                (not (effects/removable-expr? expression)))
                        host? (host-invocation-equation? available program-outputs
                                                           shape-scalars source-prefix?
                                                           equation)
                        ;; Required shape membership authorizes the exceptional one-shot branch;
                        ;; only the frontend's explicit proof contract retains a removable scalar
                        ;; as a host-only graph equation.
                        graph-shape? (true? (get-in equation
                                                    [:attributes :graph-shape-definition]))
                        ;; A preceding exceptional scalar which stays in the device equation
                        ;; sequence is an ordering barrier: a later checked shape cannot move
                        ;; ahead of it merely because invocation needs that shape.
                        state (cond-> state
                                (or (not scalar?)
                                    (and (not host?)
                                         (not (effects/removable-expr? expression))))
                                (assoc :source-prefix? false))]
                    (cond
                      (and host? exceptional-shape?)
                      ;; The checked algebra moves, with its typed closed region, into the
                      ;; InvocationPlan ScalarCompute. Removing this equation makes the value an
                      ;; explicit program input and leaves the attested plan as its sole evaluator.
                      (-> state
                          (update :hoisted conj equation)
                          (update :available into (:results equation)))

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
                                      (update :attributes assoc :host-only true
                                              :graph-shape-definition true)))
                          (update :available into (:results equation)))

                      :else
                      (update state :retained conj equation))))
                {:available (set (:inputs parallel-program))
                 :source-prefix? true :hoisted [] :retained []}
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

(defn- refine-effect-result-shapes
  "Refine the invocation boundary from the nested algorithm's storage/alias certificate.
   The algorithm retains its symbolic values and explicit iteration extent; this does not
   resize its operation or turn a physical destination capacity into a logical prefix extent."
  [values equations]
  (reduce
   (fn [values equation]
     (let [algorithm (:algorithm equation)
           facts (when (soac/program-form? algorithm) (soac/facts algorithm))]
       (reduce
        (fn [values inner]
          (let [id (second inner)
                aliases (get-in facts [:equations id :aliases])]
            (reduce
             (fn [values [result {:keys [destination access host-return]}]]
               ;; A synthesized effect result represents the mutated destination, not a newly
               ;; sized tensor. Never expand arbitrary alias/prefix views.
               (if (and (= :effect host-return) (= :read-write access)
                        (= destination (get aliases result))
                        (get values result) (get values destination))
                 (if-let [refined (av/merge-refinement (get values result) (get values destination))]
                   (assoc values result refined)
                   values)
                 values))
             values (map vector (nth inner 2) (soac/result-storage facts id)))))
        values (when facts (soac/equations algorithm)))))
   values equations))

(defn- valid-scalar-cast-value?
  "A return cast may disappear only when the typed value proves cast(T,T).

   The AbstractValue, not a source hint or operation registry entry, owns the operand dtype. It
   must be a valid effect-free rank-zero numeric tensor with a known dtype; aliases normalize
   through the canonical dtype authority before comparison."
  [value target-dtype]
  (try
    (and (= value (av/validate! value))
         (= :tensor (:kind value))
         (empty? (:shape value))
         (empty? (:effects value))
         (keyword? (:dtype value))
         (dtype/known? (:dtype value))
         (= (dtype/canon (:dtype value)) target-dtype))
    (catch Exception _ false)))

(defn- canonicalize-value-dtype
  "Normalize only a known keyword dtype for producer/outer comparison; retain every other facet."
  [value]
  (if (and (av/abstract-value? value) (keyword? (:dtype value))
           (dtype/known? (:dtype value)))
    (update value :dtype dtype/canon)
    value))

(defn- certified-result-storage
  "Return complete `{:entries [[result storage] ...] :host-binding ...}` groups from the
   validated algorithm authority.

   TypedSOAC owns storage and host binding in its inner equation facts. Generic SWR validates only
   outer storage as part of its equation boundary, so it contributes no host-binding assertion.
   Structured control currently has no result-storage projection; an outer attribute alone cannot
   manufacture one."
  [parallel-program equation]
  (let [algorithm (:algorithm equation)]
    (try
      (when (typed-algorithm-boundary? (:values parallel-program) equation algorithm)
        (cond
          (soac/program-form? algorithm)
          (let [facts (soac/facts algorithm)
                outer-results (set (:results equation))]
            (into []
                  (keep (fn [inner]
                          (let [equation-id (second inner)
                                entries (mapv vector
                                              (nth inner 2)
                                              (soac/result-storage facts equation-id))]
                            (when (some (comp outer-results first) entries)
                              {:entries entries
                               :host-binding
                               (get-in facts [:equations equation-id
                                              :attributes :host-binding])}))))
                  (soac/equations algorithm)))

          (swr/plan? algorithm)
          [{:entries (mapv vector (:results equation)
                           (get-in equation [:attributes :result-storage]))
            :host-binding nil}]

          :else []))
      (catch Exception _ nil))))

(defn- validated-produced-value
  "Return a result's AbstractValue only when its unique typed producer certifies the same value.

   The outer value table is not independently forgeable dtype authority. TypedSOAC and structured
   control retain their own result values; generic SWR's checked equation boundary relates its
   descriptor to the outer value."
  [parallel-program result]
  (let [producers (filterv #(some #{result} (:results %)) (:equations parallel-program))]
    (when (= 1 (count producers))
      (let [equation (first producers)
            algorithm (:algorithm equation)
            outer (get (:values parallel-program) result)
            authoritative
            (try
              (when (typed-algorithm-boundary? (:values parallel-program) equation algorithm)
                (cond
                  (soac/program-form? algorithm) (get-in (soac/facts algorithm) [:values result])
                  (control/loop-program? algorithm) (get (control/outer-values algorithm) result)
                  (swr/plan? algorithm) outer
                  :else nil))
              (catch Exception _ nil))]
        (when (and authoritative
                   (= (canonicalize-value-dtype outer)
                      (canonicalize-value-dtype authoritative)))
          outer)))))

(defn- validated-public-resident-output
  "Return the certified one-element resident value for an exact public reduction output.

   This requires the authoritative inner TypedSOAC certificate written by resident realization;
   an outer attribute, rank-zero value, or source `aget` spelling is never sufficient."
  [parallel-program result]
  (try
    (let [producers (filterv #(some #{result} (:results %)) (:equations parallel-program))]
      (when (= 1 (count producers))
        (let [equation (first producers)
            algorithm (:algorithm equation)
            outer (get (:values parallel-program) result)
            inner-facts (when (and (soac/program-form? algorithm)
                                   (typed-algorithm-boundary?
                                    (:values parallel-program) equation algorithm))
                          (soac/facts algorithm))
            inner-producers (when inner-facts
                              (filterv #(some #{result} (nth % 2))
                                       (soac/equations algorithm)))
            inner-producer (when (= 1 (count inner-producers))
                             (first inner-producers))
            producer-kind (some-> inner-producer soac/operation-kind)
            reduction-certified?
            (and (= 'reduce producer-kind)
                 (= {:kind :resident-scalar-buffer :elements 1}
                    (get-in inner-facts
                            [:equations (second inner-producer) :attributes
                             :resident-realization]))
                 (some #{result}
                       (get-in inner-facts
                               [:attributes :public-resident-reduction-outputs])))
            scalar-certified?
            (and (= 'scalar producer-kind)
                 (resident/validated-terminal-resident-output parallel-program result)
                 (= {:kind :resident-scalar-buffer :elements 1}
                    (get-in inner-facts [:values result :representation]))
                 (= :device (get-in inner-facts [:values result :memory-space]))
                 (= result
                    (:result (device-scalar/projection
                              (soac/equation-subprogram algorithm inner-producer)))))
            inner (when (and (= [result] (nth inner-producer 2 nil))
                             (some #{result} (soac/outputs algorithm))
                             (or reduction-certified? scalar-certified?))
                    (get-in inner-facts [:values result]))
            value (when (= (canonicalize-value-dtype outer)
                           (canonicalize-value-dtype inner))
                    outer)]
        (when (and (= :tensor (:kind value))
                   (= [] (:shape value))
                   (keyword? (:dtype value)) (dtype/known? (:dtype value))
                   (= {:kind :resident-scalar-buffer :elements 1}
                      (:representation value))
                   (= :device (:memory-space value))
                   (nil? (:logical-layout value))
                   (nil? (:placement value))
                   (nil? (:sharding value))
                   (contains? #{nil :owned} (:ownership value))
                   (empty? (:effects value))
                   (empty? (:attributes value)))
          value))))
    (catch Exception _ nil)))

(defn- source-result-projection
  "Prove one complete source return and return `{:kind ... :outputs [...]}`, or nil.

   Symbols are admitted only through typed results, certified physical destinations, exact lexical
   aliases, or effect bindings. Vectors and nil recurse structurally. A unary primitive cast is an
   identity only when retained walker identity agrees with fresh resolution in the original source
   namespace, the head is not lexically shadowed, its operand projects to exactly one result, and
   that result's validated rank-zero AbstractValue proves the same canonical numeric dtype.
   Consequently spelling, source hints, forged metadata, and non-identity casts prove nothing."
  [parallel-program {:keys [source-ns public-parameters active-params array-types]} expression]
  (let [source (:source parallel-program)
        bindings (second source)
        outputs (set (mapcat :results (:equations parallel-program)))
        aliases (into {} (keep (fn [[symbol expression]]
                                 (when (symbol? expression) [symbol expression])))
                      (partition 2 bindings))
        lexical-heads (into (set (concat public-parameters active-params))
                            (keep (fn [[symbol _]] (when (symbol? symbol) symbol)))
                            (partition 2 bindings))
        resolve-alias (fn [symbol]
                        (loop [symbol symbol seen #{}]
                          (cond
                            (contains? seen symbol) nil
                            (contains? aliases symbol) (recur (get aliases symbol)
                                                             (conj seen symbol))
                            (contains? outputs symbol) symbol
                            :else symbol)))
        equations (:equations parallel-program)
        storage-groups (mapcat (partial certified-result-storage parallel-program) equations)
        storage-entries (mapcat :entries storage-groups)
        storage-source-kind (fn [storage]
                              (case (:host-return storage)
                                :buffer :buffer
                                :effect :buffer
                                nil))
        stored-result-kinds (into {} (keep (fn [[result storage _]]
                                             (when-let [kind (storage-source-kind storage)]
                                               [result kind])))
                                  storage-entries)
        stored-results (into #{} (map first) storage-entries)
        destinations (into {} (keep (fn [[result storage _]]
                                      (when-let [kind (storage-source-kind storage)]
                                        [(:destination storage)
                                         {:output result :kind kind}])))
                           storage-entries)
        buffer-host-bindings
        (into {}
              (keep (fn [{:keys [entries host-binding]}]
                      (when (and host-binding (= 1 (count entries))
                                 (= :buffer (:host-return (second (first entries)))))
                        [host-binding (ffirst entries)])))
              storage-groups)
        effect-bindings
        (into #{}
              (keep (fn [{:keys [entries host-binding]}]
                      (when (and host-binding (seq entries)
                                 (every? #(= :effect (:host-return (second %))) entries))
                        host-binding)))
              storage-groups)
        symbol-output (fn [symbol]
                        (let [symbol (resolve-alias symbol)]
                          (cond
                            (contains? effect-bindings symbol) {:kind :unit :outputs []}
                            (contains? outputs symbol)
                            (let [kind (cond
                                         (contains? stored-result-kinds symbol)
                                         (get stored-result-kinds symbol)
                                         (contains? stored-results symbol) nil
                                         (contains? array-types symbol) :buffer
                                         :else :value)]
                              (when kind {:kind kind :outputs [symbol]}))
                            (contains? outputs (:output (get destinations symbol)))
                            {:kind (:kind (get destinations symbol))
                             :outputs [(:output (get destinations symbol))]}
                            (contains? outputs (get buffer-host-bindings symbol))
                            {:kind :buffer :outputs [(get buffer-host-bindings symbol)]}
                            :else nil)))]
    (letfn [(project [expression]
              (cond
                (symbol? expression)
                (symbol-output expression)

                (vector? expression)
                (loop [remaining expression projected []]
                  (if (seq remaining)
                    (when-let [projection (project (first remaining))]
                      (recur (next remaining) (into projected (:outputs projection))))
                    {:kind :tuple :outputs projected}))

                (nil? expression) {:kind :unit :outputs []}

                (seq? expression)
                (let [head (first expression)
                      lexical-call? (or (:raster.op/lexical-callee (meta expression))
                                        (:raster.op/lexical-callee (meta head)))
                      canonical (when-not lexical-call?
                                  (callee-identity/validated-source-callee
                                   head source-ns lexical-heads))
                      resident-output (when (and (= 'clojure.core/aget canonical)
                                                 (= 3 (count expression))
                                                 (symbol? (second expression))
                                                 (= 0 (nth expression 2)))
                                        (second expression))
                      resident-value (when (and resident-output
                                                (contains? outputs resident-output))
                                       (validated-public-resident-output
                                        parallel-program resident-output))
                      target-dtype (some-> canonical descriptor/cast-result-tag
                                           dtype/dtype-for-scalar-tag)
                      operand-projection (when (and target-dtype (= 2 (count expression)))
                                           (project (second expression)))
                      operand-output (when (and (= :value (:kind operand-projection))
                                                (= 1 (count (:outputs operand-projection))))
                                       (first (:outputs operand-projection)))
                      value (validated-produced-value parallel-program operand-output)]
                  (cond
                    resident-value {:kind :value :outputs [resident-output]}
                    (and operand-output
                         (valid-scalar-cast-value? value (dtype/canon target-dtype)))
                    operand-projection))

                :else nil))]
      (project expression))))

(defn- validate-source-result!
  "Island live-outs are not evidence that the complete source return was lowered.
   Prove the return against logical results and the existing physical-storage relation
   before attaching a whole-program invocation contract. Never recognize an operation
   by its source name to manufacture a missing continuation."
  [parallel-program {:keys [return-tag] :as options}]
  (let [source (:source parallel-program)
        _ (when-not (and (seq? source) (contains? #{'let 'let*} (first source)))
            (fail! :structured-control-invocation-source
                   "an analyzed TypedSOAC program requires a flat retained host source boundary"
                   {:source source}))
        body (drop 2 source)
        projection (when (= 1 (count body))
                     (source-result-projection parallel-program options (first body)))]
    (when-not projection
      (fail! :structured-control-source-result
             "the complete source return is not represented by the typed program outputs"
             {:source-result (last body) :source-body (vec body)
              :program-outputs (:outputs parallel-program)
              :site [:return] :source-meta (meta (last body))}))
    ;; Island outputs include host live-outs and conservative terminal values. Public
    ;; promotion changes only their escape boundary, never the equations/effects that
    ;; compute them. All returned leaves must have a typed producer or storage alias.
    (assoc parallel-program :outputs
           (if (= 'void return-tag)
             []
             (vec (distinct (:outputs projection)))))))

(defn- validate-host-coverage!
  "Host-controlled bindings may remain in an island's retained source, but a complete
   invocation must either execute them or prove them removable. Exact retained binding
   expressions, not advisory host IDs or coincident source sites, certify coverage."
  [parallel-program prefix]
  (let [prefix-bindings (into {} prefix)
        allocations (into {} (map (juxt :destination :source-expression))
                          (get-in parallel-program [:attributes :allocations]))
        claimed (into (set (concat (keys prefix-bindings) (keys allocations)))
                      (mapcat (fn [equation]
                                (concat (when (= :binding (first (:site equation)))
                                          [(second (:site equation))])
                                        (map :binding (get-in equation [:attributes :realization-bindings])))))
                      (:equations parallel-program))]
    (doseq [[symbol expression] (partition 2 (second (:source parallel-program)))
            :when (and (not (and (contains? prefix-bindings symbol)
                                 (semantic/equivalent? expression (get prefix-bindings symbol))))
                       (not (program/equation-for-binding parallel-program symbol expression))
                       (not (typed-route/realization-binding? parallel-program symbol expression))
                       (not (and (contains? allocations symbol)
                                 (semantic/equivalent? expression (get allocations symbol))))
                       ;; A changed claimed binding invalidates its executor even when the new
                       ;; source expression is pure. Only unclaimed host work may be removed.
                       (or (contains? claimed symbol)
                           (not (effects/removable-expr? expression))))]
      (fail! :structured-control-host-continuation
             "a retained host computation has no executor in the complete typed invocation"
             {:binding symbol :expression expression :site [:binding symbol]
              :source-meta (meta expression)}))
    parallel-program))

(defn promote-program
  "Attach the public invocation contract to an analyzed loop-free typed ParallelProgram.

   The numerical equations are unchanged. This pass adds only the public invocation contract:
   direct parameters map by identity, while the transitive non-equation host prefix becomes typed
   ShapeProjection/ScalarCompute/allocation steps. Structured loops and loop-free algorithms can
   therefore share scheduling, emission, ProgramCall, LinkPlan, and runtime lowering."
  [parallel-program {:keys [public-parameters active-params array-types scalar-types]
                     :as options}]
  (when-not (contains? #{:typed-soac :typed-parallel} (:dialect parallel-program))
    (fail! :structured-control-soac-promotion
           "loop-free promotion requires an analyzed typed ParallelProgram"
           {:dialect (:dialect parallel-program)}))
  (when (= :typed-parallel (:dialect parallel-program))
    (validate-typed-program! parallel-program))
  (when-let [providers (seq (get-in parallel-program [:attributes :native-initialization-providers]))]
    (fail! :structured-control-native-initialization
           "source-independent promotion cannot discard native allocation and host writes"
           {:destinations (vec providers)}))
  (let [parallel-program (-> parallel-program
                             (validate-source-result! options)
                             (assoc :dialect :typed-parallel)
                             validate-typed-program!)
        parallel-program (hoist-host-invocation-equations parallel-program)
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
        program-values (refine-effect-result-shapes (merge shape-values program-values)
                                                   (:equations parallel-program))
        parallel-program (assoc parallel-program :values program-values)
        prefix (invocation-prefix parallel-program public-parameters host-values)
        _ (validate-host-coverage! parallel-program prefix)
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
                            :algorithm-dialect (:dialect parallel-program)
                            :target-dialect :typed-invocation}})]
    (-> promoted
        (update :attributes assoc
                :host-control :typed-invocation
                :invocation-plan (invocation/validate-against! plan promoted))
        validate-typed-program!)))

(defn- source-free-dimension!
  "Validate the deliberately small dimension grammar supported by direct TypedSOAC invocation.

   This is an admission check over retained AbstractValues, not a shape inference pass.  The
   materializer remains the execution authority for these same forms."
  [public-values public-buffers public-input? owner dimension]
  (let [integral-scalar?
        (fn [id]
          (let [value (get public-values id)
                declared (:dtype value)
                dt (when (and (keyword? declared) (dtype/known? declared))
                     (dtype/canon declared))]
            (and (symbol? id) (scalar-value? value) (contains? #{:int :long} dt))))]
    (cond
      (and (integer? dimension) (<= 0 dimension Long/MAX_VALUE)) dimension

      (symbol? dimension)
      (if (integral-scalar? dimension)
        dimension
        (fail! :structured-control-source-free-dimension
               "source-free buffer shape requires a public integral scalar"
               {:value owner :dimension dimension :public-values (set (keys public-values))}))

      (and (seq? dimension) (= 'value (first dimension)) (= 2 (count dimension)))
      (let [id (second dimension)]
        (if (integral-scalar? id)
          dimension
          (fail! :structured-control-source-free-dimension
                 "source-free value dimension requires a public integral scalar"
                 {:value owner :dimension dimension :public-values (set (keys public-values))})))

      (and (seq? dimension) (= 'extent (first dimension)) (= 2 (count dimension)))
      (let [id (second dimension)]
        (if (and (contains? public-buffers id)
                 ;; Parameter materialization knows only scalar values plus its own physical
                 ;; extent. Produced values are materialized afterward and may name any public
                 ;; buffer extent through the existing concrete-shape environment.
                 (or (not public-input?) (= owner id)))
          dimension
          (fail! :structured-control-source-free-dimension
                 "source-free extent dimension requires an available public rank-one buffer"
                 {:value owner :dimension dimension :public-input? public-input?
                  :public-buffers public-buffers})))

      :else
      (fail! :structured-control-source-free-dimension
             "source-free buffer shape is outside the materializable dimension grammar"
             {:value owner :dimension dimension}))))

(defn- validate-source-free-typed-boundary!
  [typed-program]
  (let [typed-program (soac/validate! typed-program)
        facts (soac/facts typed-program)
        inputs (:inputs facts)
        public-inputs (set inputs)
        values (:values facts)
        public-values (select-keys values inputs)
        public-buffers
        (into #{}
              (keep (fn [id]
                      (let [value (get values id)]
                        (when (and (= :tensor (:kind value)) (= 1 (count (:shape value)))) id))))
              inputs)]
    (when-not (and (vector? inputs) (every? symbol? inputs)
                   (= (count inputs) (count (distinct inputs))))
      (fail! :structured-control-source-free-inputs
             "source-free TypedSOAC inputs must be ordered unique symbols"
             {:inputs inputs}))
    (doseq [id inputs
            :let [value (get values id) rank (count (:shape value))
                  declared (:dtype value)]]
      (when-not (and (= :tensor (:kind value)) (contains? #{0 1} rank)
                     (keyword? declared) (dtype/known? declared))
        (fail! :structured-control-source-free-input
               "source-free invocation requires known typed scalar and flat buffer inputs"
               {:input id :value value})))
    (when-not (and (empty? (:effects facts)) (empty? (:aliases facts))
                   (empty? (get-in facts [:attributes :native-initialization-providers])))
      (fail! :structured-control-source-free-effects
             "source-free invocation requires a pure program without native initialization"
             {:effects (:effects facts) :aliases (:aliases facts)
              :native-initialization-providers
              (get-in facts [:attributes :native-initialization-providers])}))
    (doseq [equation (soac/equations typed-program)
            :let [equation-id (second equation)
                  equation-facts (get-in facts [:equations equation-id])]]
      (when-not (and (empty? (:effects equation-facts))
                     (empty? (:aliases equation-facts))
                     (empty? (soac/result-storage facts equation-id)))
        (fail! :structured-control-source-free-equation
               "source-free invocation requires equations without effects, aliases or result storage"
               {:equation equation-id :effects (:effects equation-facts)
                :aliases (:aliases equation-facts)
                :result-storage (soac/result-storage facts equation-id)})))
    ;; Validate every retained tensor shape because scheduled physical ABIs may mention values
    ;; beyond the public input/output frontier.  A private scalar must never become an implicit
    ;; host dimension merely because it appears in an AbstractValue.
    (doseq [[id value] values
            :when (and (= :tensor (:kind value)) (seq (:shape value)))
            dimension (:shape value)]
      (source-free-dimension! public-values public-buffers (contains? public-inputs id)
                              id dimension))
    typed-program))

(defn promote-typed-algorithm
  "Promote one pure, already constructed TypedSOAC into the ordinary typed invocation boundary.

   Public parameters are exactly the algorithm's ordered inputs and retain their complete
   AbstractValues.  There is no source prefix, inferred type override, external result storage or
   host expression.  The initial boundary is deliberately limited to scalar and rank-one inputs
   whose retained shapes are directly materializable from public integral scalars or public buffer
   extents.  Scheduling and emission remain separate ordinary compiler stages."
  [typed-program]
  (let [typed-program (validate-source-free-typed-boundary! typed-program)
        facts (soac/facts typed-program)
        inputs (:inputs facts)
        outputs (soac/outputs typed-program)
        values (:values facts)
        parallel-program (-> (typed-route/program-envelope typed-program)
                             (assoc :dialect :typed-parallel)
                             (update :provenance assoc
                                     :front-end :explicit-typed-algorithm
                                     :pass :source-free-typed-invocation))
        plan (invocation/from-prefix
              {:id [:source-free-typed-invocation (semantic/fingerprint typed-program)]
               :parameters inputs
               :parameter-values (select-keys values inputs)
               :bindings []
               :binding-values {}
               :program-values values
               :program-inputs inputs
               :program-storage []
               :program-outputs outputs
               :attributes {:source-dialect :typed-soac
                            :algorithm-dialect :typed-soac
                            :target-dialect :typed-invocation}})
        promoted (update parallel-program :attributes assoc
                         :host-control :typed-invocation
                         :invocation-plan (invocation/validate-against! plan parallel-program))]
    (validate-typed-program! promoted)))

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
           :algorithm? (partial typed-algorithm-boundary? values)})
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

               (swr/plan? algorithm)
               (let [graph (:graph (equation-graph/make-for-plan-equation
                                    parallel-program equation))
                     descriptor (or (:target-descriptor opts)
                                    (when (map? (:target-device opts)) (:target-device opts))
                                    {})
                     certificate
                     (try
                       (let [schedule-for-node
                             (case (get-in opts [:schedule :segmented-weighted-reduction :strategy]
                                      :auto)
                          ;; Explicit subgroup selection authorizes the candidate's stated dot
                          ;; reassociation. Auto stays reference until equation dispatch carries
                          ;; per-alternative preflight and numerical-policy-aware tuning evidence.
                          :subgroup-score-reuse indexed-body/schedule-score-reuse-for-node
                          (:auto :reference) indexed-body/schedule-reference-for-node)]
                         (if (contains? opts :scalar-math)
                           (schedule-for-node algorithm (first (:nodes graph)) graph descriptor
                                              (select-keys opts [:scalar-math]))
                           (schedule-for-node algorithm (first (:nodes graph)) graph descriptor)))
                       (catch clojure.lang.ExceptionInfo exception
                         (if (= :indexed-segmented-reduction-plan-unsupported
                                (:reason (ex-data exception)))
                           (fail! :segmented-weighted-reduction-unscheduled
                                  "segmented weighted-reduction has no verified equation-first schedule"
                                  {:equation (:id equation) :site (:site equation)
                                   :plan-id (:id algorithm) :decline (ex-data exception)})
                           (throw exception))))]
                 {:equations (conj equations (assoc equation :operations [certificate]))
                  :values values})

               :else
               (let [algorithm-values (:values (soac/facts algorithm))
                     schedule-options
                     (assoc opts
                            :device-scalar-context parallel-program
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
       :operation? scheduled-operation?}))))
