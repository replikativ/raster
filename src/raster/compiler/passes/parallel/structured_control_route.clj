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
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as invocation-source]
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
  [parallel-program equation algorithm]
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
         (boolean (swr/equation-boundary? (:values parallel-program) equation algorithm
                                           parallel-program)))

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
    (and (swr/equation-boundary? (:values parallel-program) equation algorithm parallel-program)
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
                     (partial typed-algorithm-boundary? parallel-program)))

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

(defn- scalar-value?
  [value]
  (and (= :tensor (:kind value)) (empty? (:shape value))))

(defn- project-scalar-equation-metadata
  [program-values public-inputs equation]
  (if-let [expression (invocation-source/scalar-equation-expression equation)]
    (let [projected (invocation-source/project-scalar-metadata program-values public-inputs expression)]
      (if (semantic/retained-source-equivalent? expression projected)
        equation
        (let [result (first (:results equation))
              scalar-types (into {} (keep (fn [[id value]]
                                            (when (scalar-value? value) [id (:dtype value)])))
                                 program-values)
              algorithm (:algorithm equation)
              operation (typed-frontend/scalar-equation
                         {:id (second (first (soac/equations algorithm)))
                          :sym result :expr projected} scalar-types scalar-types)
              captures (:captures (soac/operation-parts operation))
              facts (assoc (soac/facts algorithm) :values program-values :inputs captures)
              algorithm (soac/validate! (soac/make facts [operation] [result]))]
          (assoc equation :algorithm algorithm :operations [operation] :operands captures))))
    equation))

(defn- host-invocation-equation?
  [available program-outputs required-shape-scalars source-prefix? equation]
  (let [expression (invocation-source/scalar-equation-expression equation)]
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
  (let [source-allocation-arguments
        (mapcat (fn [{:keys [extent source-expression]}]
                  (concat (util/free-syms extent) (util/free-syms source-expression)))
                (get-in parallel-program [:attributes :allocations]))
        shape-symbols
        (into #{}
              (concat source-allocation-arguments
                      (mapcat (fn [value] (mapcat util/free-syms (:shape value)))
                              (vals (:values parallel-program)))))
        definitions
        (into {}
              (mapcat (fn [equation]
                        (when (invocation-source/scalar-equation-expression equation)
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
  [parallel-program public-inputs]
  (let [program-outputs (:outputs parallel-program)
        shape-scalars (required-shape-scalars parallel-program)
        {:keys [hoisted retained]}
        (reduce (fn [{:keys [available source-prefix?] :as state} equation]
                  (let [original-equation equation
                        equation (project-scalar-equation-metadata (:values parallel-program)
                                                                  public-inputs equation)
                        expression (invocation-source/scalar-equation-expression equation)
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
                      (update state :retained conj original-equation))))
                {:available (set/union (set (:inputs parallel-program)) public-inputs)
                 :source-prefix? true :hoisted [] :retained []}
                (:equations parallel-program))]
    (if (and (empty? hoisted) (= retained (:equations parallel-program)))
      parallel-program
      (let [inputs (program/infer-inputs retained)
            shape-equations (count (filter (comp invocation-source/shape-projection-source
                                                 invocation-source/scalar-equation-expression)
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
                              (mapcat (fn [symbol]
                                        (let [expression (get definitions symbol)]
                                          (if-let [source (invocation-source/reference-identity-source
                                                           symbol expression)]
                                            [source]
                                            (util/free-syms expression)))))
                              set
                              (set/intersection available))
            required' (set/union required dependencies)]
        (if (= required required') required (recur required'))))))

(defn- exact-source-owner?
  "Existing source execution ownership, not a purity or completion claim."
  [parallel-program symbol expression]
  (or (program/equation-for-binding parallel-program symbol expression)
      (invocation-source/realization-binding? parallel-program symbol expression)
      (some (fn [{:keys [destination source-expression]}]
              (and (= symbol destination)
                   (semantic/retained-source-equivalent? expression source-expression)))
            (get-in parallel-program [:attributes :allocations]))))

(defn- invocation-binding-expression
  "Preserve allocation recipes and their lexical dependencies; normalize only scalar projections.
   Allocation size equivalence is checked separately in the causal InvocationPlan environment."
  [values public-inputs symbol expression]
  (if (invocation-source/allocation-source expression values)
    expression
    (invocation-source/canonicalize-shape-projection values public-inputs symbol expression)))

(defn- validate-retained-source-owners!
  "Existing source claims must each remain valid before considering another executor.
   A matching prefix, equation or allocation may not conceal another stale claim."
  [parallel-program]
  (let [allocations (get-in parallel-program [:attributes :allocations])
        _ (when-not (= (count allocations) (count (set (map :destination allocations))))
            (fail! :structured-control-host-continuation
                   "allocation destinations require unique retained source ownership"
                   {:owner :allocation :destinations (mapv :destination allocations)}))
        definitions (group-by first (partition 2 (second (:source parallel-program))))
        actual (fn [binding]
                 (let [pairs (get definitions binding)]
                   (when (= 1 (count pairs)) (second (first pairs)))))
        check! (fn [binding valid? family]
                 (when-not valid?
                   (fail! :structured-control-host-continuation
                          "retained execution ownership differs from its source definition"
                          {:binding binding :site [:binding binding] :owner family})))]
    (doseq [equation (:equations parallel-program)]
      (when (= :binding (first (:site equation)))
        (let [binding (second (:site equation))]
          (check! binding (and (some? (actual binding))
                               (semantic/retained-source-equivalent? (:source equation) (actual binding)))
                  :equation)))
      (doseq [{:keys [binding]} (get-in equation [:attributes :realization-bindings])]
        (check! binding (= (:id equation)
                           (:id (invocation-source/realization-equation
                                 parallel-program binding (actual binding))))
                :realization)))
    (doseq [{:keys [destination source-expression]} (get-in parallel-program [:attributes :allocations])]
      (check! destination (and (some? (actual destination))
                              (semantic/retained-source-equivalent?
                               source-expression (actual destination)))
              :allocation)))
  parallel-program)

(defn- retain-source-allocation-values
  "Complete only missing outer source-allocation contracts from their causal definitions.
   Numerical initialization has already run; these values add no equation, input or output."
  [program host-values]
  (reduce
   (fn [program [allocation-index {:keys [destination source-expression extent dtype initialization]
                                  :as allocation}]]
     (if-not (or (contains? allocation :source-expression)
                 (contains? allocation :source-binding-id))
       program
       (let [value (get host-values destination)
             existing (get-in program [:values destination])]
         ;; Claim validation is independent of insertion. Re-entering promotion
         ;; with a completed outer value cannot conceal changed claim metadata.
         (when-not (and (= :tensor (:kind value)) (= 1 (count (:shape value)))
                        (or (nil? dtype) (= dtype (:dtype value)))
                        (or (nil? existing) (= (:dtype existing) (:dtype value)))
                        (semantic/retained-source-equivalent? extent (first (:shape value)))
                        (invocation-source/allocation-corresponds?
                         source-expression value initialization host-values))
           (fail! :structured-control-source-allocation-value
                  "source-only allocation requires its exact constructor-owned buffer contract"
                  {:allocation allocation :value value}))
         (-> (cond-> program (nil? existing) (assoc-in [:values destination] value))
             (assoc-in [:attributes :allocations allocation-index :dtype] (:dtype value))))))
   program (map-indexed vector (get-in program [:attributes :allocations]))))

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
          raw-source-pairs (mapv vec (partition 2 (second source)))
          source-pairs (mapv (fn [[symbol expression]]
                               [symbol (invocation-binding-expression
                                        prefix-values public-inputs symbol expression)])
                             raw-source-pairs)
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
          retained-identities (into #{} (keep (fn [[symbol expression]]
                                               (when (invocation-source/preserved-reference-identity?
                                                      symbol expression) symbol))) candidates)
          ;; This is an ordered step catalog, not permission to evaluate a whole prefix early.
          ;; Retained observable work must execute or refuse, even when functionalization removes
          ;; its numerical use. Already-owned auxiliary actions retain their existing executor.
          retained-actions (into #{} (keep (fn [[symbol expression]]
                                             (when (and (not (exact-source-owner? parallel-program symbol expression))
                                                        (not (effects/removable-expr? expression)))
                                               symbol))) raw-source-pairs)
          roots (set/difference (set/union (set (:inputs parallel-program)) shape-roots
                                           retained-identities retained-actions)
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

   TypedSOAC owns storage and host binding in its inner equation facts. Generic SWR additionally
   authenticates the retained protected source before contributing a host-binding assertion.
   Structured control currently has no result-storage projection; an outer attribute alone cannot
   manufacture one."
  [parallel-program equation]
  (let [algorithm (:algorithm equation)]
    (try
      (when (typed-algorithm-boundary? parallel-program equation algorithm)
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
          [(or (swr/source-buffer-return parallel-program equation)
               {:entries (mapv vector (:results equation)
                               (get-in equation [:attributes :result-storage]))
                :host-binding nil})]

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
              (when (typed-algorithm-boundary? parallel-program equation algorithm)
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

(defn ^:no-doc validated-public-resident-output
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
                                   (or (typed-algorithm-boundary?
                                        parallel-program equation algorithm)
                                       (scheduled-algorithm-boundary?
                                        parallel-program equation algorithm)))
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

(defn- certified-source-storage
  "One source-return view of the existing validated result-storage groups.
   Logical effect results name storage but do not return host buffer references."
  [parallel-program]
  (let [groups (vec (mapcat (partial certified-result-storage parallel-program)
                            (:equations parallel-program)))
        buffer-host-entries (keep (fn [{:keys [entries host-binding]}]
                                    (when (and host-binding (= 1 (count entries))
                                               (= :buffer (:host-return (second (first entries)))))
                                      [host-binding (first entries)])) groups)]
    {:entries (mapcat :entries groups)
     :buffer-host-bindings (into {} (map (fn [[binding [result _ _]]] [binding result])) buffer-host-entries)
     :buffer-reference-sources (into {} (map (fn [[binding [_ storage _]]]
                                              [binding (:destination storage)])) buffer-host-entries)
     :effect-bindings (into #{}
                            (keep (fn [{:keys [entries host-binding]}]
                                    (when (and host-binding (seq entries)
                                               (every? #(= :effect (:host-return (second %))) entries))
                                      host-binding))) groups)}))

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
        {:keys [entries buffer-host-bindings effect-bindings]}
        (certified-source-storage parallel-program)
        storage-entries entries
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
  ([parallel-program prefix public-inputs]
   (validate-host-coverage! parallel-program prefix public-inputs (:values parallel-program)))
  ([parallel-program prefix public-inputs projection-values]
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
                                 (semantic/retained-source-equivalent?
                                  (invocation-binding-expression
                                   projection-values
                                   public-inputs
                                   symbol expression)
                                  (get prefix-bindings symbol))))
                       (not (exact-source-owner? parallel-program symbol expression))
                       ;; A changed claimed binding invalidates its executor even when the new
                       ;; source expression is pure. Only unclaimed host work may be removed.
                       (or (contains? claimed symbol)
                           (not (effects/removable-expr? expression))))]
      (fail! :structured-control-host-continuation
             "a retained host computation has no executor in the complete typed invocation"
             {:binding symbol :expression expression :site [:binding symbol]
              :source-meta (meta expression)}))
    parallel-program)))

(defn promote-program
  "Attach the public invocation contract to an analyzed source-bearing typed ParallelProgram.

   The numerical equations are unchanged. This pass adds only the public invocation contract:
   direct parameters map by identity, while non-equation host dependencies and unowned observable
   source actions become ordered ShapeProjection/ScalarCompute/allocation steps. The step catalog
   is not permission to evaluate late actions before numerical execution. Structured loops and
   loop-free algorithms share this same boundary and preserve one authoritative source order."
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
                             validate-retained-source-owners!
                             (assoc :dialect :typed-parallel)
                             validate-typed-program!)
        public-parameters (vec (or public-parameters active-params))
        parallel-program (hoist-host-invocation-equations parallel-program
                                                         (set public-parameters))
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
        source-storage (certified-source-storage parallel-program)
        host-analysis
        (host-av/analyze-capacities
          (:source parallel-program)
          (assoc options :values parameter-values :lexical-inputs public-parameters
                 :array-types array-types :scalar-types scalar-types
                 :non-buffer-bindings (:effect-bindings source-storage)
                 :reference-aliases
                 (invocation-source/capacity-reference-aliases
                  (partition 2 (second (:source parallel-program))) public-parameters
                  (:buffer-reference-sources source-storage))))
        ;; Final relational representatives can name a LATER alength binding.
        ;; Invocation capacities instead use each source definition's original
        ;; point-in-time contract; numerical program shapes stay independent.
        host-values (:definition-values host-analysis)
        parallel-program (retain-source-allocation-values parallel-program host-values)
        program-values (:values parallel-program)
        ;; An unknown physical extent may still be refined by its source-owned
        ;; definition. Use only the causal snapshot, never final relational
        ;; representatives, and leave every known numerical domain unchanged.
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
        projection-values (merge host-values program-values)
        prefix (invocation-prefix parallel-program public-parameters projection-values)
        ;; Construction and coverage must project in the identical causal environment. Host-only
        ;; allocation aliases need not occur in the numerical value table; omitting them here
        ;; would compare a normalized extent with the original alength expression instead.
        _ (validate-host-coverage! parallel-program prefix (set public-parameters)
                                  projection-values)
        [binding-values _]
        (reduce (fn [[resolved available] [symbol expression]]
                     (let [identity-source (invocation-source/reference-identity-source symbol expression)
                           ;; A dead identity may have no numerical equation/AV.
                           ;; It still denotes exactly one preceding invocation
                           ;; value; use that existing contract, not array hints.
                           identity-value (when identity-source (get available identity-source))
                           _ (when (and identity-source (nil? identity-value))
                               (fail! :structured-control-prefix-identity-source
                                      "reference identity requires a preceding invocation value"
                                      {:symbol symbol :source identity-source}))
                           program-value (get program-values symbol)
                           host-value (or identity-value (get host-values symbol))
                           value (cond
                                   ;; An exact lexical alias keeps one runtime storage identity.
                                   ;; Its program-side symbolic shape remains authoritative and
                                   ;; materialization later proves both symbolic extents resolve to
                                   ;; the same concrete view before binding.
                                   (and program-value host-value (or (symbol? expression) identity-source))
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
                       [(assoc resolved symbol value) (assoc available symbol value)]))
              [{} parameter-values]
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
  ([public-values public-buffers public-input? owner dimension]
   (source-free-dimension! public-values public-buffers {} public-input? owner dimension))
  ([public-values public-buffers projected-values public-input? owner dimension]
  (let [integral-scalar-in?
        (fn [values id]
          (let [value (get values id)
                declared (:dtype value)
                dt (when (and (keyword? declared) (dtype/known? declared))
                     (dtype/canon declared))]
            (and (symbol? id) (scalar-value? value) (contains? #{:int :long} dt))))
        integral-scalar?
        (fn [id]
          (or (integral-scalar-in? public-values id)
              (integral-scalar-in? projected-values id)))]
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
        (if (integral-scalar-in? public-values id)
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
             {:value owner :dimension dimension})))))

(defn- validate-source-free-value-boundary!
  ([typed-program] (validate-source-free-value-boundary! typed-program [] []))
  ([typed-program shape-projections external-dimensions]
  (let [typed-program (soac/validate! typed-program)
        facts (soac/facts typed-program)
        inputs (:inputs facts)
        public-inputs (set inputs)
        values (:values facts)
        public-values (select-keys values inputs)
        projected-values (select-keys values
                                      (concat (map :dimension shape-projections)
                                              (map :id external-dimensions)))
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
    ;; Validate every retained tensor shape because scheduled physical ABIs may mention values
    ;; beyond the public input/output frontier.  A private scalar must never become an implicit
    ;; host dimension merely because it appears in an AbstractValue.
    (doseq [[id value] values
            :when (and (= :tensor (:kind value)) (seq (:shape value)))
            dimension (:shape value)]
      (source-free-dimension! public-values public-buffers projected-values
                              (contains? public-inputs id)
                              id dimension))
    typed-program)))

(defn- validate-source-free-typed-boundary!
  ([typed-program] (validate-source-free-typed-boundary! typed-program [] []))
  ([typed-program shape-projections external-dimensions]
  (let [typed-program (validate-source-free-value-boundary!
                       typed-program shape-projections external-dimensions)
        facts (soac/facts typed-program)]
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
    typed-program)))

(defn- source-free-invocation-plan
  [kind typed-program shape-projections external-dimensions external-storage extra-attributes]
  (let [facts (soac/facts typed-program)
        inputs (:inputs facts)
        values (:values facts)
        parameter-values
        (reduce (fn [parameter-values {:keys [source]}]
                  (update parameter-values source assoc :shape [(list 'extent source)]))
                (select-keys values inputs) shape-projections)
        bindings (mapv (fn [{:keys [dimension source]}]
                         [dimension (list 'clojure.core/alength source)])
                       shape-projections)
        attributes
        (cond-> (merge {:source-dialect :typed-soac
                        :algorithm-dialect :typed-soac
                        :target-dialect :typed-invocation}
                       extra-attributes)
          (seq shape-projections)
          (assoc :source-free-shape-projections (vec shape-projections))
          (seq external-dimensions)
          (assoc :external-dimensions :resident-premise
                 :external-dimension-values (vec external-dimensions)))]
    (invocation/from-prefix
     {:id [kind (semantic/fingerprint typed-program)]
      :parameters inputs :parameter-values parameter-values
      :bindings bindings
      :binding-values (select-keys values (map :dimension shape-projections))
      :program-values values :program-inputs inputs
      :external-program-storage external-storage
      :program-outputs (soac/outputs typed-program)
      :attributes attributes})))

(defn promote-typed-algorithm
  "Promote one pure, already constructed TypedSOAC into the ordinary typed invocation boundary.

   Public parameters are exactly the algorithm's ordered inputs. A rank-one input whose sole
   dimension is one private retained integral scalar may project that dimension from its own host
   array extent; the program AbstractValue remains unchanged and every other parameter facet is
   exact. There is no inferred type override, external result storage or host expression.
   Scheduling and emission remain separate ordinary compiler stages."
  [typed-program]
  (let [typed-program (soac/validate! typed-program)
        parallel-program (-> (typed-route/program-envelope typed-program)
                             (assoc :dialect :typed-parallel)
                             (update :attributes assoc :host-control :typed-invocation)
                             (update :provenance assoc
                                     :front-end :explicit-typed-algorithm
                                     :pass :source-free-typed-invocation))
        shape-projections (invocation/source-free-shape-projections parallel-program)
        external-dimensions (invocation/source-free-external-dimensions
                             parallel-program shape-projections)
        typed-program (validate-source-free-typed-boundary!
                       typed-program shape-projections external-dimensions)
        plan (source-free-invocation-plan
              :source-free-typed-invocation typed-program
              shape-projections external-dimensions [] {})
        promoted (update parallel-program :attributes assoc
                         :invocation-plan (invocation/validate-against! plan parallel-program))]
    (validate-typed-program! promoted)))

(defn promote-typed-effect-algorithm
  "Promote the strict sequential ordered effect-map subset with unresolved resident storage.

   Public parameters remain only the algorithm's declared logical inputs. The sole physical
   destination is derived from validated result-storage and retained as an unresolved external
   premise: this function neither allocates nor initializes it. Host materialization refuses this
   plan; a resident binder and final LinkPlan must separately prove compatible storage and ordered
   initialization."
  [typed-program]
  (let [typed-program (soac/validate! typed-program)
        parallel-program (-> (typed-route/program-envelope typed-program)
                             (assoc :dialect :typed-parallel)
                             (update :attributes assoc :host-control :typed-invocation)
                             (update :provenance assoc
                                     :front-end :explicit-typed-effect-algorithm
                                     :pass :source-free-typed-effect-invocation))
        shape-projections (invocation/source-free-shape-projections parallel-program)
        external-dimensions (invocation/source-free-external-dimensions
                             parallel-program shape-projections)
        typed-program (validate-source-free-value-boundary!
                       typed-program shape-projections external-dimensions)
        facts (soac/facts typed-program)
        _ (when (seq (get-in facts [:attributes :native-initialization-providers]))
            (fail! :structured-control-source-free-effects
                   "source-free external storage cannot carry native initialization"
                   {:native-initialization-providers
                    (get-in facts [:attributes :native-initialization-providers])}))
        {:keys [destination]} (typed-route/source-free-ordered-effect-storage! typed-program)
        plan (source-free-invocation-plan
              :source-free-typed-effect-invocation typed-program
              shape-projections external-dimensions [destination]
              {:external-storage :resident-premise})
        promoted (update parallel-program :attributes assoc
                         :invocation-plan (invocation/validate-against! plan parallel-program))]
    (validate-typed-program! promoted)))

(defn promote-source-free-algorithm
  "Select the unchanged pure promotion or the separately checked ordered-effect promotion."
  [typed-program]
  (let [typed-program (soac/validate! typed-program)
        facts (soac/facts typed-program)]
    (if (and (empty? (:effects facts)) (empty? (:aliases facts))
             (every? #(empty? (soac/result-storage facts (second %)))
                     (soac/equations typed-program)))
      (promote-typed-algorithm typed-program)
      (promote-typed-effect-algorithm typed-program))))

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

(defn- scope-generated-results [typed scope]
  (let [renames (into {} (keep (fn [id] (when-not (symbol? id) [id [scope id]])))
                      (mapcat #(nth % 2) (soac/equations typed)))]
    (if (seq renames) (soac/remap-values typed renames) typed)))

(defn- prefix-program
  [{:keys [prefix-bindings parameter-values loop loop-binding loop-source
           suffix-bindings outer-body]} {:keys [abstract-machine] :as options}]
  (when (seq prefix-bindings)
    (let [options (-> options
                      (assoc :values parameter-values)
                      (update :array-types #(merge (value-types parameter-values seq) %))
                      (update :scalar-types #(merge (value-types parameter-values empty?) %)))
          options (if-let [parameters (or (:public-parameters options) (:active-params options))]
                    (assoc options :lexical-inputs parameters) options)
          source (list 'let* (vec (mapcat identity prefix-bindings)) nil)
          analysis (typed-frontend/source-fragment-analysis source options)]
      (when-let [last-numerical (last (:numerical-binding-ids analysis))]
        (let [cutoff (inc last-numerical)
              head (subvec prefix-bindings 0 cutoff)
              tail (subvec prefix-bindings cutoff)
              continuation (list* 'let*
                                  (vec (mapcat identity
                                               (concat tail [[loop-binding loop-source]] suffix-bindings)))
                                  outer-body)
              needed (into (set (control/outer-operands loop)) (util/free-syms continuation))
              outputs (filterv needed (mapv first head))
              source (list 'let* (vec (mapcat identity head)) outputs)
              typed (typed-frontend/form->program source options)]
          (when-not typed
            (fail! :structured-control-unsupported-prefix
                   "pre-loop numerical work is outside the certified TypedSOAC subset"
                   {:prefix-bindings head}))
          (let [fused (-> (first (fusion/fusion-fixpoint typed abstract-machine))
                          (scope-generated-results :prefix))
                envelope (typed-route/analyzed-program-envelope source fused)]
            {:envelope envelope :typed fused :tail tail :cutoff cutoff
             :source-attributes (:attributes analysis)}))))))

(defn- prefix-storage-results [typed]
  (let [facts (soac/facts (soac/validate! typed))]
    (reduce
     (fn [mapping equation]
       (reduce (fn [mapping [result {:keys [destination host-return]}]]
                 (if (and (= :buffer host-return)
                          (= destination (get-in facts [:equations (second equation) :aliases result])))
                   (do (when (contains? mapping destination)
                         (fail! :structured-control-prefix-storage
                                "a prefix storage join requires one preceding result owner"
                                {:destination destination}))
                       (assoc mapping destination result))
                   mapping))
               mapping (map vector (nth equation 2) (soac/result-storage facts (second equation)))))
     {} (soac/equations typed))))

(defn- prefix-loop-join [{:keys [loop outer-values outer-equalities]} prefix]
  (if-not prefix
    {:dimension-loop loop :loop loop :dimensions {}}
    (let [values (get-in prefix [:envelope :values])
          results (prefix-storage-results (:typed prefix))
          incoming (set (control/outer-operands loop))
          available (into (set (get-in prefix [:envelope :inputs]))
                          (mapcat :results (get-in prefix [:envelope :equations])))
          buffers (select-keys results incoming)
          analysis {:values outer-values :equalities outer-equalities}
          dimensions
          (reduce-kv
           (fn [renames physical logical]
             (let [before (get-in (control/outer-values loop) [physical :shape])
                   after (:shape (get values logical))]
               (when-not (= (count before) (count after))
                 (fail! :structured-control-prefix-shape
                        "a prefix result must preserve the complete loop input rank"
                        {:physical physical :logical logical :before before :after after}))
               (reduce (fn [renames [old new]]
                         (if (= old new) renames
                           (do
                             (when-not (and (contains? incoming old)
                                            (contains? available new)
                                            (= [] (:shape (get outer-values old)))
                                            (contains? #{:int :long} (:dtype (get outer-values old)))
                                            (= [] (:shape (get values new)))
                                            (contains? #{:int :long} (:dtype (get values new)))
                                            (= (host-av/shape-extent analysis old)
                                               (host-av/shape-extent analysis new))
                                            (or (not (contains? renames old)) (= new (get renames old))))
                               (fail! :structured-control-prefix-shape
                                      "loop dimensions need preceding source equality and one target"
                                      {:physical physical :old old :new new}))
                             (assoc renames old new))))
                       renames (map vector before after))))
           {} buffers)
          dimension-loop (control/rebind-incoming loop dimensions values)]
      {:dimension-loop dimension-loop :dimensions dimensions
       :loop (control/rebind-incoming dimension-loop buffers values)})))

(defn- suffix-prefix-values
  "Transport only unchanged preceding SSA results, retaining their exact numerical contracts.
   Source capacity equality is checked independently; it never rewrites the producer's shape."
  [{:keys [outer-values outer-equalities suffix-bindings outer-body]} prefix loop]
  (let [values (get-in prefix [:envelope :values])
        lexical-values (into util/*shadowing-locals*
                             (concat (keys values) (keys (control/outer-values loop))))
        referenced (binding [util/*shadowing-locals* lexical-values]
                     (util/free-syms (list* 'let* (vec (mapcat identity suffix-bindings)) outer-body)))
        loop-values (set (concat (control/outer-operands loop) (control/outer-results loop)))
        selected (remove loop-values (filter referenced (get-in prefix [:envelope :outputs])))
        analysis {:values outer-values :equalities outer-equalities}]
    (doseq [id selected
            :let [producer (get values id) source (get outer-values id)]]
      (when-not (and source
                     (= (dissoc producer :shape) (dissoc source :shape))
                     (= (count (:shape producer)) (count (:shape source)))
                     (= (mapv #(host-av/shape-extent analysis %) (:shape producer))
                        (mapv #(host-av/shape-extent analysis %) (:shape source))))
        (fail! :structured-control-prefix-suffix-value
               "suffix entry must preserve one preceding producer and its source storage contract"
               {:value id :producer producer :source source})))
    (clojure.core/loop [pending (seq selected) result {}]
      (if-let [id (first pending)]
        (if (contains? result id)
          (recur (next pending) result)
          (let [producer (get values id)
                loop-value (get (control/outer-values loop) id)
                _ (when (and loop-value (not= producer loop-value))
                    (fail! :structured-control-prefix-suffix-value
                           "a producer shape dependency cannot replace a loop entry contract"
                           {:value id :producer producer :loop-value loop-value}))
                value (or loop-value producer)]
            (when-not value
              (fail! :structured-control-prefix-suffix-value
                     "a preceding producer shape requires a declared dependency" {:value id}))
            (recur (concat (next pending)
                           (binding [util/*shadowing-locals* lexical-values]
                             (doall (mapcat util/free-syms (:shape value)))))
                   (assoc result id value))))
        result))))

(defn- suffix-program
  [{:keys [loop suffix-bindings outer-body outer-values]}
   {:keys [dtype abstract-machine] :as options} incoming-values]
  (when (or (seq suffix-bindings) (seq outer-body))
    (let [source (list* 'let* (vec (mapcat identity suffix-bindings)) outer-body)
          ;; Analyze the source's physical carry names first. Fresh loop results belong to the
          ;; enclosing SSA boundary, not this original source environment.
          outputs (control/outer-results loop)
          references (binding [util/*shadowing-locals*
                               (into util/*shadowing-locals*
                                     (concat (keys outer-values) (keys incoming-values)
                                             (keys (control/outer-values loop))))]
                       (util/free-syms source))
          values (apply dissoc
                        (merge (select-keys outer-values references)
                               (control/outer-values loop)
                               ;; Earlier numerical definitions own their SSA contracts. The
                               ;; source host-capacity snapshot is not a second numerical producer
                               ;; and may use another (equivalent) extent spelling.
                               incoming-values)
                        outputs)
          typed (typed-frontend/form->program
                 source (assoc options
                               :dtype (or dtype :double)
                               :values values
                               :array-types (apply dissoc (merge (:array-types options)
                                                                (value-types values seq)) outputs)
                               :scalar-types (apply dissoc (merge (:scalar-types options)
                                                                 (value-types values empty?)) outputs)))]
      (when-not typed
        (fail! :structured-control-unsupported-suffix
               "post-loop computation is outside the certified TypedSOAC subset"
               {:suffix-bindings suffix-bindings :outer-body outer-body}))
      (let [fused (scope-generated-results
                   (first (fusion/fusion-fixpoint typed abstract-machine)) :suffix)
            inputs (set (:inputs (soac/facts fused)))
            substitutions (into {} (keep (fn [{:keys [initial output]}]
                                            (when (contains? inputs initial) [initial output])))
                                (control/carried loop))]
        {:envelope (typed-route/analyzed-program-envelope source fused substitutions)
         :source source :typed fused :incoming-renames substitutions}))))

(defn- offset-source-facts
  "Rebase outer source provenance; the nested algorithm keeps its own local source frame."
  [facts offset]
  (cond-> facts
    (integer? (get-in facts [:provenance :source-binding-id]))
    (update-in [:provenance :source-binding-id] + offset)
    (seq (get-in facts [:attributes :fusion/constituents]))
    (update-in [:attributes :fusion/constituents]
               #(into {} (map (fn [[id facts]]
                                [(if (integer? id) (+ id offset) id)
                                 (offset-source-facts facts offset)])) %))))

(defn- offset-source-attributes [attributes offset]
  (let [shift #(cond-> % (integer? (:source-binding-id %)) (update :source-binding-id + offset))]
    (cond-> attributes
      (:host-binding-ids attributes) (update :host-binding-ids #(mapv (partial + offset) %))
      (:host-read-sites attributes) (update :host-read-sites #(mapv shift %))
      (:allocations attributes) (update :allocations #(mapv shift %)))))

(defn- composed-source-attributes [prefix suffix prefix-bindings loop-binding]
  (let [head (get-in prefix [:envelope :attributes])
        complete (:source-attributes prefix)
        cutoff (:cutoff prefix)
        tail-hosts (filterv #(>= % cutoff) (:host-binding-ids complete))
        tail-reads (filterv #(>= (:source-binding-id %) cutoff) (:host-read-sites complete))
        before (if prefix
                 (assoc (merge head complete)
                        :host-binding-ids (into (vec (:host-binding-ids head)) tail-hosts)
                        :host-read-sites (into (vec (:host-read-sites head)) tail-reads))
                 {})
        after (offset-source-attributes (:attributes suffix) (inc (count prefix-bindings)))]
    (-> (merge before after)
        (assoc :source-bindings (into (conj (mapv first prefix-bindings) loop-binding)
                                     (get-in suffix [:attributes :source-bindings])))
        (assoc :host-binding-ids (into (vec (:host-binding-ids before)) (:host-binding-ids after))
               :host-read-sites (into (vec (:host-read-sites before)) (:host-read-sites after))
               :host-read-values (vec (distinct (concat (:host-read-values before)
                                                        (:host-read-values after))))
               :allocations (into (vec (:allocations before)) (:allocations after))))))

(defn- normalize-decomposition [decomposition options]
  (if-not (:source decomposition)
    decomposition
    (let [options (update options :array-types
                          #(merge (value-types (:parameter-values decomposition) seq) %))
          options (if-let [parameters (or (:public-parameters options) (:active-params options))]
                    (assoc options :lexical-inputs parameters) options)
          source (typed-frontend/normalize-source (:source decomposition) options)]
      (if (identical? source (:source decomposition)) decomposition
        (or (frontend/form->structured-loop source options)
            (fail! :structured-control-normalized-source
                   "normalized structured source must retain its verified loop construction" {}))))))

(defn- loop-equation
  [loop loop-binding loop-source]
  (let [facts (control/facts loop)]
    (program/->ProgramEquation
     [:structured-loop (:id facts)] [:binding loop-binding] loop-source
     (ordered-distinct (control/outer-operands loop))
     (control/outer-results loop) loop [loop]
     (:effects facts)
     (assoc (:provenance facts) :pass :structured-control-route)
     (assoc (:attributes facts) :algorithm-dialect :typed-structured-control))))

(defn- assemble-program-envelope
  [{:keys [source loop-binding loop-source prefix-bindings]} prefix loop suffix-envelope]
  (let [loop-equation (loop-equation loop loop-binding loop-source)
         prefix-equations (mapv #(update % :id (fn [id] [:prefix id]))
                               (get-in prefix [:envelope :equations]))
         source-offset (inc (count prefix-bindings))
         suffix-equations (mapv #(-> (offset-source-facts % source-offset)
                                     (update :id (fn [id] [:suffix id])))
                                (:equations suffix-envelope))
         source (if (and source suffix-envelope)
                  (with-meta (list* (first source)
                                    (vec (mapcat identity
                                                 (concat (if prefix
                                                           (concat (partition 2 (second (get-in prefix [:envelope :source])))
                                                                   (:tail prefix))
                                                           prefix-bindings)
                                                         [[loop-binding loop-source]]
                                                         (partition 2 (second (:source suffix-envelope))))))
                                    (drop 2 (:source suffix-envelope)))
                             (meta source))
                  source)
         equations (into (conj prefix-equations loop-equation) suffix-equations)
         values (merge-values (merge-values (get-in prefix [:envelope :values] {})
                                            (control/outer-values loop))
                              (:values suffix-envelope))
         program-outputs (or (:outputs suffix-envelope) (control/outer-results loop))
         {:keys [inputs outputs]} (program-boundary equations program-outputs)
         effects (reduce set/union #{} (map :effects equations))
         parallel-program (program/make
          {:dialect :typed-parallel
           :source source
           :values values
           :inputs inputs
           :equations equations
           :outputs outputs
           :effects effects
           :provenance {:source-dialect :typed-structured-control
                        :pass :structured-control-route}
           :attributes (assoc (composed-source-attributes prefix suffix-envelope prefix-bindings loop-binding)
                               :host-control :typed-structured-control
                               :mixed-algorithms (boolean suffix-envelope))
           :operation? typed-operation?
           :algorithm? (partial typed-algorithm-boundary?
                                {:values values :inputs inputs :equations equations})})]
    parallel-program))

(defn- realize-source-return-reductions
  "Demand resident representation only after the complete assembled source return is proved.
   Rebuild through the same source/envelope constructor; never copy representation certificates
   onto the outer program. Source-free fragments and prefix live-outs acquire no demand."
  [decomposition prefix loop suffix-result options]
  (let [envelope (:envelope suffix-result)
        initial (assemble-program-envelope decomposition prefix loop envelope)
        source-owned? (and (:source decomposition) (:resident-reductions? options)
                           (seq (or (:public-parameters options) (:active-params options))))]
    (if-not suffix-result
      initial
      (let [projected (when source-owned? (validate-source-result! initial options))
            ;; Projection uses the enclosing SSA names, while this retained source fragment
            ;; still owns its original incoming names. Results are unchanged by that transport.
            returned (set (:outputs projected))
            typed (:typed suffix-result)
            demand (if source-owned? (resident/source-return-reduction-demand typed returned) [])]
          (let [optimized (typed-route/realize-typed-component typed options demand)
                _ (when-let [decline (:declined optimized)]
                    (fail! :structured-control-unsupported-suffix
                           "source-return representation has no supported typed realization"
                           {:decline decline}))
                rebuilt (typed-route/analyzed-program-envelope
                         (:source suffix-result) (:program optimized)
                         (:incoming-renames suffix-result))]
            (assemble-program-envelope decomposition prefix loop rebuilt))))))

(defn program-envelope
  "Wrap one certified frontend decomposition in the shared typed program envelope.

   When a suffix is itself in the TypedSOAC subset, its uses of the physical carried array are
   alpha-remapped to the loop's fresh logical output and appended as ordinary typed equations."
  ([decomposition] (program-envelope decomposition {}))
  ([decomposition options]
   (frontend/validate-decomposition! decomposition)
   (let [{:keys [source prefix-bindings parameter-values outer-values]
          :as decomposition} (normalize-decomposition decomposition options)
         prefix (when source (prefix-program decomposition options))
         {:keys [loop dimension-loop dimensions]} (prefix-loop-join decomposition prefix)
         suffix-result (suffix-program
                        (assoc decomposition :loop dimension-loop)
                        (if prefix (assoc options :shape-equalities dimensions) options)
                        (when prefix (suffix-prefix-values decomposition prefix dimension-loop)))
         parallel-program (realize-source-return-reductions decomposition prefix loop suffix-result options)
         {:keys [values inputs outputs]} parallel-program
         public-parameters (or (:public-parameters options) (:active-params options))]
     (if (and (seq public-parameters) source)
       (promote-program parallel-program options)
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
         parallel-program)))))

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
