(ns raster.compiler.passes.parallel.scheduled-equation-graph
  "Derive one target-neutral KernelGraph from a scheduled TypedSOAC program.

   This is the shared graph boundary for an ordinary equation and for one iteration of structured
   control. It consumes only the retained functional algorithm, ordered SegOps, and AbstractValue
   contracts; source spelling and operation-family names play no role."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.product-reduction-regions :as product-regions]
            [raster.compiler.passes.parallel.map-read-requirements :as map-reads]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :scheduled-equation-graph))))

(declare integral-scalar-value?)

(defn- value-elements
  [values derived-scalars value]
  (let [dimension-value (fn [dimension]
                          ;; AbstractValue uses `(value id)` to distinguish a compound stable
                          ;; value ID from shape syntax. KernelGraph owns explicit integer
                          ;; expressions, so remove only that marker at this physical boundary.
                          (if (and (seq? dimension)
                                   (= 'value (first dimension))
                                   (= 2 (count dimension)))
                            (second dimension)
                            (if (and (seq? dimension)
                                     (not (contains? '#{extent unknown-dimension} (first dimension))))
                              (let [decline! (fn [rule message data] (fail! rule message data))
                                    scope (set (filter #(integral-scalar-value? (get values %))
                                                       (util/free-syms dimension)))
                                    ;; This is AbstractValue dimension algebra, not a host
                                    ;; scalar expression. Its operations are mathematical extents
                                    ;; (scan's n+1), so no source arithmetic dtype is inferred.
                                    index (index-expression/lower dimension scope decline!)]
                                (index-expression/to-launch-expression index decline!))
                              dimension)))
        shape (mapv dimension-value (:shape value))
        elements (cond
                   (empty? shape) 1
                   (= 1 (count shape)) (first shape)
                   :else (apply launch/product shape))]
    ;; Shapes retain stable SSA identities.  A preceding pure scalar equation may define one of
    ;; those identities as checked launch algebra (notably normalised `rstr_extent_n` values).
    ;; Rebind that compiler-owned definition here, where graph allocation becomes concrete, so a
    ;; KernelBody's source-derived storage certificate and the graph describe the same extent.
    (launch/rebind-expression elements derived-scalars)))

(defn- integral-scalar-value?
  [value]
  (and value
       (empty? (:shape value))
       (contains? #{:int :long} (some-> (:dtype value) dtype/canon))))

(defn- unknown-shape? [value]
  (some #(and (seq? %) (= 'unknown-dimension (first %))) (:shape value)))

(defn- unresolved-capacity? [id value elements]
  ;; An analyzed array parameter can explicitly name its runtime allocation extent. That is
  ;; not an independent minimum shape requirement; a proven read/write requirement replaces it.
  (or (unknown-shape? value) (= elements (list 'extent id))))

(defn- required-write-extent
  "Cover every logical write to shared physical storage; an unknown write prevents refinement."
  [values derived-scalars result-values]
  (when (and (seq result-values) (every? #(and % (not (unknown-shape? %))) result-values))
    (let [extents (vec (distinct (map #(value-elements values derived-scalars %) result-values)))]
      (if (= 1 (count extents)) (first extents) (apply launch/maximum extents)))))

(defn- scalar-result-expression
  "Return one closed typed scalar result with lambda parameters replaced by its stable captures.

   This consumes the retained TypedSOAC scalar equation, rather than source spelling or an
   emitter-side symbol table.  Locals are expanded in their already-validated SSA order; a value
   that is not a pure launch expression is simply not a graph-storage definition."
  [equation]
  (let [algorithm (:algorithm equation)]
    (when (and (soac/program-form? algorithm)
               (= 1 (count (soac/equations algorithm))))
      (let [semantic-equation (first (soac/equations algorithm))
            {:keys [kind captures lambda]} (soac/operation-parts semantic-equation)
            results (nth semantic-equation 2)
            {:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
        (when (and (= 'scalar kind)
                   (= results (:results equation))
                   (= (:operands equation) (:inputs (soac/facts algorithm)))
                   (= (:results equation) (soac/outputs algorithm))
                   (= 1 (count results))
                   (= 1 (count body-results))
                   (= (count captures) (count parameters)))
          (let [substitutions
                (reduce (fn [bindings {:keys [id init]}]
                          (assoc bindings id (util/subst-syms bindings init)))
                        (zipmap parameters captures)
                        locals)]
            [(first results)
             (util/subst-syms substitutions (first body-results))]))))))

(defn- launch-definition
  [values bindings equation]
  (when-let [[result expression] (scalar-result-expression equation)]
    (let [{:keys [captures]} (soac/operation-parts
                              (first (soac/equations (:algorithm equation))))]
      (when (and (integral-scalar-value? (get values result))
                 (every? #(integral-scalar-value? (get values %)) captures))
        (try
          (let [narrowing-cast?
                (boolean
                 (some (fn [form]
                         (and (seq? form) (= 2 (count form))
                              (contains? '#{int clojure.core/int} (first form))))
                       (tree-seq coll? seq expression)))
                _ (when narrowing-cast?
                    (throw (ex-info "launch/storage scalar projection cannot erase narrowing"
                                    {:reason :derived-scalar-narrowing
                                     :result result :expression expression})))
                decline! (fn [rule message data]
                           (throw (ex-info message
                                           (assoc data :reason rule
                                                       :pass :scheduled-equation-graph))))
                index (index-expression/lower-typed
                       expression (set captures)
                       #(get-in values [% :dtype]) (get-in values [result :dtype]) decline!)
                projected (launch/rebind-expression
                           (index-expression/to-launch-expression index decline!) bindings)
                projected-dtype
                (launch/typed-expression-dtype
                 projected #(some-> (get values %) :dtype))]
            (when-not (= projected-dtype (dtype/canon (get-in values [result :dtype])))
              (throw (ex-info "launch/storage scalar projection changes its retained width"
                              {:reason :derived-scalar-dtype
                               :result result :projected projected
                               :projected-dtype projected-dtype
                               :result-dtype (get-in values [result :dtype])})))
            [result projected])
          ;; Scalar equations also represent ordinary arithmetic.  Only the monotone, integral
          ;; subset accepted by KernelLaunch is an allocation definition; other scalar work stays
          ;; opaque at this boundary exactly as before.
          (catch clojure.lang.ExceptionInfo _ nil))))))

(defn derived-scalar-expressions
  "Derive replayable graph-storage expressions from ordered, already-validated scalar equations.

   Definitions are expanded in program order, so a later extent never remains an opaque alias of
   an earlier one.  This is deliberately a small projection from TypedSOAC scalar equations to
   KernelLaunch, not a source re-parser or a new scalar registry."
  [values equations]
  (reduce (fn [bindings equation]
            (if-let [[result expression] (launch-definition values bindings equation)]
              (assoc bindings result expression)
              bindings))
          {} equations))

(defn- contiguous-equation-region!
  [parallel-program equations]
  (let [all-equations (:equations parallel-program)
        positions (into {} (map-indexed (fn [index equation] [(:id equation) index]) all-equations))
        indices (mapv #(get positions (:id %)) equations)
        first-index (first indices)
        last-index (last indices)
        span (when (and first-index last-index)
               (subvec all-equations first-index (inc last-index)))
        numerical (filterv #(not (true? (get-in % [:attributes :host-only]))) span)
        host-gap (filterv #(true? (get-in % [:attributes :host-only])) span)
        available-before (into (set (:inputs parallel-program))
                               (mapcat :results)
                               (subvec all-equations 0 (or first-index 0)))
        hoistable?
        (when (= numerical equations)
          (:ok
           (reduce (fn [{:keys [available] :as state} equation]
                     (if (and (:ok state)
                              (empty? (:effects equation))
                              (set/subset? (set (:operands equation)) available))
                       (-> state
                           (update :available into (:results equation)))
                       (assoc state :ok false)))
                   {:ok true :available available-before}
                   host-gap)))]
    (when-not (and (seq equations)
                   (every? some? indices)
                   (or (= 1 (count indices)) (apply < indices))
                   (= equations numerical)
                   hoistable?)
      (fail! :scheduled-equation-region
             "a scheduled equation region must be an exact numerical slice with only hoistable scalar gaps"
             {:equations (mapv :id equations) :indices indices
              :span (mapv :id span) :host-gap (mapv :id host-gap)}))
    {:indices indices :host-gap host-gap}))

(defn- region-outputs
  [parallel-program equations indices]
  (let [selected-ids (set (map :id equations))
        later-equations (drop (inc (last indices)) (:equations parallel-program))
        escaping (set/union (set (:outputs parallel-program))
                            (into #{} (mapcat :operands) later-equations))
        terminal-results (set (:results (peek equations)))]
    (into []
          (filter #(or (contains? escaping %) (contains? terminal-results %)))
          (mapcat :results equations))))

(defn- slice-value-contracts
  "Remove a later scalar from a slice-local tensor capacity.

  ParallelProgram keeps one value table for the whole program.  A later map can refine a shared
  input/output buffer to its normalized extent, but an earlier scheduled equation must not acquire
  that later scalar as an ABI argument.  Restore the ordinary runtime allocation-capacity marker:
  graph construction replaces it with any independently proved read/write extent, while a public
  input resolves it from the bound buffer.  Unlike `unknown-dimension`, `(extent id)` therefore
  remains bindable when an operation deliberately consumes the caller's complete allocation."
  [values equations]
  (let [inputs (set (program/infer-inputs equations))
        results (set (mapcat :results equations))
        available (set/union inputs results)
        scalar-id? (fn [id]
                     (and (contains? values id)
                          (empty? (:shape (get values id)))
                          (contains? #{:int :long}
                                     (some-> (get-in values [id :dtype]) dtype/canon))))
        closed? (fn [value]
                  (let [shape-scalars (into #{}
                                            (filter scalar-id?)
                                            (mapcat util/free-syms (:shape value)))]
                    (set/subset? shape-scalars available)))]
    (reduce-kv
     (fn [sliced id value]
       (assoc sliced id (if (and (= :tensor (:kind value))
                                 (not (closed? value)))
                          (assoc value :shape [(list 'extent id)])
                          value)))
     {} values)))

(defn body-for-equations
  "Return the dependency-closed scheduled program slice for a numerical equation region.

   The equations must be an exact contiguous slice. Earlier host-scalar definitions are retained
   as proof terms, while only terminal or escaping numerical results remain on the region boundary.
   This is the graph-level seam used by schedules that refine several semantic equations into one
   kernel; it does not itself authorize fusion or change their operations."
  [parallel-program equations]
  (let [parallel-program (program/validate! parallel-program)
        equations (vec equations)
        {:keys [indices host-gap]} (contiguous-equation-region! parallel-program equations)
        first-index (first indices)
        preceding (subvec (:equations parallel-program) 0 first-index)
        scalar-prefix (vec (filter #(true? (get-in % [:attributes :host-only])) preceding))
        outputs (region-outputs parallel-program equations indices)
        body-equations (vec (concat scalar-prefix host-gap equations))
        values (slice-value-contracts (:values parallel-program) body-equations)]
    (program/make
     {:dialect :segop
      :source nil
      :values values
      :inputs (program/infer-inputs body-equations)
      :equations body-equations
      :outputs outputs
      :effects (reduce set/union #{} (map :effects body-equations))
      :diagnostics []
      :provenance {:source-dialect :typed-soac
                   :pass :scheduled-equation-graph}
      :attributes {:host-control :explicit-typed-algorithm
                   :equation-region (mapv :id equations)}
      :operation? segop/segop-node?
      :algorithm? (fn [candidate algorithm]
                    (and (= algorithm (soac/validate! algorithm))
                         (= (:operands candidate) (:inputs (soac/facts algorithm)))
                         (= (:results candidate) (soac/outputs algorithm))))})))

(defn body-for-equation
  "Return the dependency-closed scheduled program slice for one numerical equation.

   A graph is emitted one equation at a time, but storage extents may be defined by preceding
   typed host-scalar equations.  This is the single narrowing operation used by equation-first
   target emission and compatibility backend entry; neither may discard or reconstruct that
   prefix independently."
  [parallel-program equation]
  (body-for-equations parallel-program [equation]))

(defn- typed-host-scalar-equation?
  [values equation algorithm]
  (when-let [[result _] (scalar-result-expression equation)]
    (let [semantic-equation (first (soac/equations algorithm))
          {:keys [attributes captures]} (soac/operation-parts semantic-equation)]
      (and (= algorithm (soac/validate! algorithm))
           (= 1 (count (:results equation)))
           (= (:dtypes attributes) [(get-in values [result :dtype])])
           (empty? (get-in values [result :shape]))
           (every? #(contains? values %) captures)))))

(defn- physical-outputs
  [algorithm]
  (let [facts (soac/facts algorithm)
        producers (into {}
                        (mapcat (fn [equation]
                                  (map vector (nth equation 2)
                                       (soac/physical-results facts equation))))
                        (soac/equations algorithm))]
    (mapv (fn [result]
            (or (get producers result)
                (fail! :scheduled-equation-result-storage
                       "a scheduled result has no physical storage identity"
                       {:result result})))
          (soac/outputs algorithm))))

(defn- external-inputs
  [operations]
  (:inputs
   (reduce (fn [{:keys [initialized] :as state} operation]
             (let [reads (segop/operation-inputs operation)
                   writes (segop/operation-outputs operation)]
               (-> state
                   (update :inputs set/union (set/difference reads initialized))
                   (update :initialized set/union writes))))
           {:inputs #{} :initialized #{}}
           operations)))

(defn- storage-spec
  [values derived-scalars id]
  (let [value (get values id)]
    (when-not value
      (fail! :scheduled-equation-storage-value
             "scheduled storage lacks an AbstractValue"
             {:value id}))
    {:dtype (:dtype value)
     :elements (value-elements values derived-scalars value)
     :memory-space (or (:memory-space value) :device)}))

(defn- product-read-requirements
  "Optional minimum capacities from typed product loads, never guessed from loop size alone."
  [values operations derived-scalars]
  (reduce
   (fn [requirements operation]
     (if (and (reduction/product-reduction? (:reduction operation))
              (:element (:reduction operation))
              (every? (fn [id]
                        (let [value (get values id)]
                          (and (= {:kind :plain} (:representation value))
                               (nil? (:logical-layout value)))))
                      (:inputs operation)))
       (let [derived
             (try
               (product-regions/dense-read-requirements
                operation
                {:array-types (into {} (map (fn [[id v]] [id (:dtype v)])) values)
                 :scalar-types (into {} (keep (fn [[id v]]
                                               (when (empty? (:shape v))
                                                 [id (:dtype v)]))) values)}
                (fn [rule message data] (fail! rule message data)))
               ;; This is an optional refinement. Unsupported source access stays unknown;
               ;; production admission must independently require a successful access proof.
               (catch clojure.lang.ExceptionInfo _ nil))]
         (reduce-kv (fn [result id extent]
                      (update result id (fnil conj [])
                              (launch/rebind-expression extent derived-scalars)))
                    requirements derived))
       requirements))
   {} operations))

(defn- algorithm-boundary?
  [equation algorithm]
  (and (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))

(defn- ordered-distinct
  [values]
  (reduce (fn [result value]
            (if (some #(= value %) result) result (conj result value)))
          [] values))

(defn- public-scalars
  [scheduled operations buffer-specs]
  (let [operation-required (reduce set/union #{} (map segop/operation-scalars operations))
        ;; KernelLaunch intentionally permits a compound list identity such as `(extent input)`
        ;; as one resolver-owned leaf.  It is not a graph scalar and must not be destructured or
        ;; promoted to a fictitious ABI argument.  Expanded allocation algebra, on the other
        ;; hand, closes only over stable symbol/keyword scalar identities.
        storage-required
        (into #{}
              (filter #(or (symbol? %) (keyword? %)))
              (reduce set/union #{}
                      (map #(launch/expression-references (:elements %))
                           (vals buffer-specs))))
        required (set/union operation-required storage-required)
        ordered (ordered-distinct
                 (concat (filter required (:inputs scheduled))
                         (sort-by pr-str (remove (set (:inputs scheduled)) required))))
        values (:values scheduled)]
    (mapv (fn [id]
            (let [value (get values id)]
              (when-not value
                (fail! :scheduled-equation-scalar-value
                       "scheduled scalar lacks an AbstractValue"
                       {:value id}))
              (when-not (empty? (:shape value))
                (fail! :scheduled-equation-scalar-shape
                       "scheduled scalar dependency must have scalar shape"
                       {:value id :shape (:shape value)}))
              (when (and (contains? storage-required id)
                         (not (integral-scalar-value? value)))
                (fail! :scheduled-equation-storage-scalar
                       "graph storage algebra must close over a declared integral scalar"
                       {:value id :value-contract value}))
              (graph/scalar id (:dtype value))))
          ordered)))

(defn make
  "Build a verified graph from `algorithm` and its fully scheduled SegOp ParallelProgram.

   Optional facts describe the enclosing control/equation context without changing dataflow."
  ([algorithm scheduled] (make algorithm scheduled {}))
  ([algorithm scheduled {:keys [effects provenance attributes]
                         :or {provenance {} attributes {}}}]
   (let [algorithm (soac/validate! algorithm)
         scheduled (program/validate! scheduled segop/segop-node? algorithm-boundary?)
         _ (when-not (= :segop (:dialect scheduled))
             (fail! :scheduled-equation-dialect
                    "KernelGraph derivation requires a fully scheduled :segop program"
                    {:dialect (:dialect scheduled)}))
         host-prefix (vec (take-while #(true? (get-in % [:attributes :host-only]))
                                      (:equations scheduled)))
         numerical-equations (vec (drop (count host-prefix) (:equations scheduled)))
         _ (when-not (and (seq numerical-equations)
                          (every? #(typed-host-scalar-equation? (:values scheduled) % (:algorithm %))
                                  host-prefix))
             (fail! :scheduled-equation-prefix
                    "a graph body requires an earlier-only typed host-scalar prefix and numerical equations"
                    {:host-prefix (mapv :id host-prefix)
                     :numerical-equations (mapv :id numerical-equations)}))
         retained-equations (vec (mapcat (comp soac/equations :algorithm) numerical-equations))
         ;; A scheduled graph may retain several numerical equations after its host-scalar
         ;; prefix. Infer the boundary of the complete numerical slice; using only the first
         ;; equation silently loses external operands first introduced by a later equation.
         inferred-numerical-inputs (program/infer-inputs numerical-equations)
         algorithm-inputs (:inputs (soac/facts algorithm))
         ;; Operand order in a scheduled equation can differ from the semantic program boundary
         ;; (structured control makes this visible). Compare the complete inferred boundary as a
         ;; set while retaining the already-validated TypedSOAC order as the canonical interface.
         numerical-inputs (vec (filter (set inferred-numerical-inputs) algorithm-inputs))
         _ (when-not (and (= (soac/equations algorithm) retained-equations)
                          (= (set inferred-numerical-inputs) (set algorithm-inputs))
                          (= numerical-inputs algorithm-inputs)
                          (= (soac/outputs algorithm) (:outputs scheduled)))
             (fail! :scheduled-equation-algorithm
                    "scheduled equations or program boundary differ from the retained algorithm"
                    {:algorithm-inputs algorithm-inputs
                     :scheduled-inputs inferred-numerical-inputs
                     :algorithm-outputs (soac/outputs algorithm)
                     :scheduled-outputs (:outputs scheduled)}))
         operations (vec (mapcat :operations numerical-equations))
         _ (when (empty? operations)
             (fail! :scheduled-equation-empty
                    "a KernelGraph requires at least one scheduled operation" {}))
         derived-scalars (derived-scalar-expressions (:values scheduled) host-prefix)
         values (:values scheduled)
         map-read-options
         {:array-types (into {} (map (fn [[id v]] [id (:dtype v)])) values)
          :scalar-types (into {} (keep (fn [[id v]]
                                        (when (empty? (:shape v)) [id (:dtype v)]))) values)
          :scalar-definitions derived-scalars}
         ;; Proof metadata belongs to the scheduled node, not the semantic SegOp. Mutating the
         ;; operation would change structural identity inside enclosing structured-control IR.
         ;; A target may consume this witness only after revalidating the exact node and graph.
         read-capacity-certificates
         (mapv #(map-reads/symbolic-read-certificate % map-read-options) operations)
         inputs (external-inputs operations)
         outputs (set (physical-outputs algorithm))
         operation-values (reduce set/union #{}
                                  (map #(set/union (segop/operation-inputs %)
                                                   (segop/operation-outputs %))
                                       operations))
         temporary-ids (set/difference operation-values inputs outputs)
         result-storage-values
         (reduce
          (fn [by-storage equation]
            (reduce (fn [by-storage [logical physical]]
                      (update by-storage physical (fnil conj [])
                              (get-in (soac/facts algorithm) [:values logical])))
                    by-storage
                    (map vector (nth equation 2) (soac/physical-results algorithm equation))))
          {} retained-equations)
         buffer-specs (into {}
                            (map (fn [id] [id (storage-spec values derived-scalars id)]))
                            (set/union inputs outputs temporary-ids))
         read-requirements
         (reduce
          (fn [requirements [operation certificate]]
            (if (and (some (fn [id]
                             (unresolved-capacity? id (get values id)
                                                   (get-in buffer-specs [id :elements])))
                           (:inputs operation))
                     (every? (fn [id]
                          (let [value (get values id)]
                            (and (= {:kind :plain} (:representation value))
                                 (nil? (:logical-layout value)))))
                        (:inputs operation)))
              (let [derived (or (:requirements certificate)
                                (map-reads/static-read-requirements
                                 operation map-read-options))]
                (merge-with into requirements
                            (into {} (map (fn [[id extent]] [id [extent]])) derived)))
              requirements))
          (product-read-requirements values operations derived-scalars)
          (map vector operations read-capacity-certificates))
         buffer-specs (reduce-kv
                       (fn [specs id requirements]
                         (let [extents (vec (distinct
                                             (cond-> requirements
                                               (not (unresolved-capacity?
                                                     id (get values id) (get-in specs [id :elements])))
                                               (conj (get-in specs [id :elements])))))
                               required (if (= 1 (count extents)) (first extents)
                                            (apply launch/maximum extents))]
                           (assoc-in specs [id :elements] required)))
                       buffer-specs read-requirements)
         ;; An aliased destination can have unknown capacity while the typed result has a
         ;; precise written extent (e.g. an exclusive scan writes n+1 elements). That semantic
         ;; extent is the required capacity, not a claim about the allocation's full size.
         buffer-specs (reduce-kv
                       (fn [specs id result-values]
                         (if-let [extent (and (contains? specs id)
                                              (unresolved-capacity?
                                               id (get values id)
                                               (:elements (storage-spec values derived-scalars id)))
                                              (required-write-extent values derived-scalars result-values))]
                           (assoc-in specs [id :elements]
                                     (if (contains? read-requirements id)
                                       (let [read-extent (get-in specs [id :elements])]
                                         (if (= read-extent extent) extent
                                             (launch/maximum read-extent extent)))
                                       extent))
                           specs))
                       buffer-specs result-storage-values)
         capacity-preconditions
         (->> read-capacity-certificates
              (mapcat (comp seq :requirements))
              (keep (fn [[id required]]
                      (let [capacity (get-in buffer-specs [id :elements])]
                        (when-not (map-reads/capacity-covers? capacity required)
                          {:expression capacity :op :>= :value required}))))
              distinct
              vec)
         scalars (public-scalars scheduled operations buffer-specs)]
     (let [kernel-graph
           (graph/from-segops
            operations
            {:inputs inputs
             :outputs outputs
             :temporaries (select-keys buffer-specs temporary-ids)
             :scalars scalars
             :preconditions capacity-preconditions
             :buffer-specs buffer-specs
             :dtype (:dtype (first (vals buffer-specs)))
             :effects (or effects {:semantic (:effects (soac/facts algorithm))})
             :provenance (merge {:source-dialect :typed-soac
                                 :algorithm-dialect :typed-soac
                                 :schedule-dialect :segop}
                                provenance)
             ;; Derived scalar definitions are deliberately absent here.  They are proof terms
             ;; in the retained ParallelProgram prefix, not descriptive graph attributes.
             :attributes attributes})
           nodes (mapv (fn [node certificate]
                         (cond-> node certificate
                           (assoc :read-capacity-certificate certificate)))
                       (:nodes kernel-graph) read-capacity-certificates)]
       (graph/validate! (assoc kernel-graph :nodes nodes))))))

(defn validate-projection!
  "Require the exact graph projection of an independently retained algorithm and scheduled body.
   Descriptive graph context is preserved, but cannot supply storage or scalar definitions."
  [kernel-graph algorithm scheduled-body]
  (when-not (and algorithm scheduled-body)
    (fail! :scheduled-equation-projection-context
           "graph projection requires both retained algorithm and scheduled body" {}))
  (let [expected (make algorithm scheduled-body
                       (select-keys kernel-graph [:effects :provenance :attributes]))]
    (when-not (= expected kernel-graph)
      (fail! :scheduled-equation-projection
             "graph differs from its retained algorithm and scheduled body"
             {:expected expected :actual kernel-graph})))
  kernel-graph)

(defn algorithm-for-equations
  "Compose the exact retained TypedSOAC algorithms for a contiguous scheduled equation region.

   Equation forms and equation facts are copied from the independently validated per-equation
   algorithms. The enclosing ParallelProgram supplies only the authoritative value table and
   boundary analysis; no source form is reparsed and no numerical operation is synthesized."
  [parallel-program equations]
  (let [parallel-program (program/validate! parallel-program)
        equations (vec equations)
        {:keys [indices]} (contiguous-equation-region! parallel-program equations)
        algorithms (mapv (comp soac/validate! :algorithm) equations)
        equation-forms (vec (mapcat soac/equations algorithms))
        equation-facts (apply merge (map (comp :equations soac/facts) algorithms))
        inputs (program/infer-inputs equations)
        outputs (region-outputs parallel-program equations indices)
        effects (reduce set/union #{} (map (comp :effects soac/facts) algorithms))]
    (soac/make
     (soac/default-program-facts
      {:values (:values parallel-program)
       :inputs inputs
       :equations equation-facts
       :effects effects
       :provenance {:source-dialect :typed-soac
                    :pass :scheduled-equation-region}
       :attributes {:equation-region (mapv :id equations)}})
     equation-forms outputs)))

(defn make-for-equations
  "Derive one exact scheduled KernelGraph for a contiguous TypedSOAC equation region.

   This constructs the unfused semantic source graph. A later schedule may replace that graph with
   one kernel only through a checked ScheduledGraphRefinement that preserves its public boundary."
  [parallel-program equations]
  (let [equations (vec equations)
        body (body-for-equations parallel-program equations)
        algorithm (algorithm-for-equations parallel-program equations)]
    {:algorithm algorithm
     :body body
     :graph (make algorithm body)}))

(defn make-for-equation
  "Derive the exact scheduled KernelGraph for one TypedSOAC numerical equation.

   The equation's optional graph contract contributes only already-validated schedule provenance
   and attributes; dataflow, storage and scalar closure are always reconstructed from the retained
   algorithm and its dependency-closed program slice."
  [parallel-program equation]
  (let [body (body-for-equation parallel-program equation)
        algorithm (:algorithm equation)
        graph-contract (get-in equation [:attributes :kernel-graph])]
    {:body body
     :graph (make algorithm body
                  (cond-> {}
                    graph-contract
                    (assoc :provenance (:provenance graph-contract)
                           :attributes (:attributes graph-contract))))}))
