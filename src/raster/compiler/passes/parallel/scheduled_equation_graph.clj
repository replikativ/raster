(ns raster.compiler.passes.parallel.scheduled-equation-graph
  "Derive one target-neutral KernelGraph from a scheduled TypedSOAC program.

   This is the shared graph boundary for an ordinary equation and for one iteration of structured
   control. It consumes only the retained functional algorithm, ordered SegOps, and AbstractValue
   contracts; source spelling and operation-family names play no role."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-closure :as contraction-closure]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.passes.parallel.product-reduction-regions :as product-regions]
            [raster.compiler.passes.parallel.typed-soac-projection :as typed-projection]
            [raster.compiler.passes.parallel.map-read-requirements :as map-reads]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :scheduled-equation-graph))))

(defn- value-elements
  [values derived-scalars value]
  (let [dimension-value (fn [dimension]
                          (index-expression/project-dimension
                           dimension #(get-in values [% :dtype])
                           (fn [rule message data] (fail! rule message data))))
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

(defn- dependency-closed-body
  [parallel-program equations
   {:keys [dialect source-dialect operation? algorithm? body-attributes]}]
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
     {:dialect dialect
      :source nil
      :values values
      :inputs (program/infer-inputs body-equations)
      :equations body-equations
      :outputs outputs
      :effects (reduce set/union #{} (map :effects body-equations))
      :diagnostics []
      :provenance {:source-dialect source-dialect
                   :pass :scheduled-equation-graph}
      :attributes (merge {:host-control :explicit-typed-algorithm
                          :equation-region (mapv :id equations)}
                         body-attributes)
      :operation? operation?
      :algorithm? algorithm?})))

(defn body-for-equations
  "Return the dependency-closed scheduled program slice for a numerical equation region.

   The equations must be an exact contiguous slice. Earlier host-scalar definitions are retained
   as proof terms, while only terminal or escaping numerical results remain on the region boundary.
   This is the graph-level seam used by schedules that refine several semantic equations into one
   kernel; it does not itself authorize fusion or change their operations."
  [parallel-program equations]
  (dependency-closed-body
   parallel-program equations
   {:dialect :segop
    :source-dialect :typed-soac
    :operation? segop/segop-node?
    :algorithm? (fn [candidate algorithm]
                  (and (= algorithm (soac/validate! algorithm))
                       (= (:operands candidate) (:inputs (soac/facts algorithm)))
                       (= (:results candidate) (soac/outputs algorithm))))}))

(defn body-for-equation
  "Return the dependency-closed scheduled program slice for one numerical equation.

   A graph is emitted one equation at a time, but storage extents may be defined by preceding
   typed host-scalar equations.  This is the single narrowing operation used by equation-first
   target emission and compatibility backend entry; neither may discard or reconstruct that
   prefix independently."
  [parallel-program equation]
  (body-for-equations parallel-program [equation]))

(defn- typed-host-scalar-equation-for-validated-algorithm?
  ;; Only called after make validates the enclosing scheduled program with algorithm-boundary?.
  ;; That check already validates this exact retained algorithm; keep the prefix-specific dtype,
  ;; scalar-result and capture checks here without deriving its complete SOAC proof twice.
  [values equation algorithm]
  (when-let [[result _] (scalar-result-expression equation)]
    (let [semantic-equation (first (soac/equations algorithm))
          {:keys [attributes captures]} (soac/operation-parts semantic-equation)]
      (and (= 1 (count (:results equation)))
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

(defn- contraction-read-requirements
  "Project independent typed operand maps, including stage and epilogue operands."
  [algorithm equations]
  (reduce
   (fn [requirements equation]
     (let [{:keys [kind attributes arrays captures]} (soac/operation-parts equation)]
       (cond
         (= 'contract kind)
         (let [bindings (contraction-closure/bindings attributes arrays captures)]
           (reduce (fn [result {:keys [parameter elements]}]
                     (update result (get bindings parameter) (fnil conj []) elements))
                   requirements (contraction-closure/storage-requirements attributes)))

         (= 'segmented-reduce kind)
         (let [reads (try
                       (typed-projection/segmented-reduce-core-read-requirements algorithm equation)
                       ;; Optional logical-domain refinement, never physical address admission.
                       (catch clojure.lang.ExceptionInfo _ nil))]
           (reduce-kv (fn [result id minimum]
                        (update result id (fnil conj []) minimum)) requirements (or reads {})))

         :else requirements)))
   {} equations))

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
                          (every? #(typed-host-scalar-equation-for-validated-algorithm?
                                    (:values scheduled) % (:algorithm %))
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
         transform-read-requirements
         (into {}
               (map (fn [equation]
                      [(second equation)
                       (reduce
                        (fn [requirements {:keys [value] operand-map :map}]
                          (let [minimum (value-elements values derived-scalars
                                                        {:shape (axis-map/shape operand-map)})]
                            (update requirements value
                                    (fn [prior]
                                      (if (or (nil? prior) (= prior minimum)) minimum
                                          (launch/maximum prior minimum))))))
                        {} (get-in (soac/operation-parts equation)
                                   [:attributes :result-transform :operands]))]))
               retained-equations)
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
            (let [marked-inputs
                  (set/intersection
                   (set (:inputs operation))
                   (set (mapcat #(get-in (soac/operation-parts %)
                                        [:attributes :attributes :pointwise-storage-inputs])
                                (filter #(= (second %) (:algorithm-equation operation))
                                        retained-equations))))]
            (if (and (or (seq marked-inputs)
                         (some (fn [id]
                             (unresolved-capacity? id (get values id)
                                                   (get-in buffer-specs [id :elements])))
                           (:inputs operation)))
                     (every? (fn [id]
                          (let [value (get values id)]
                            (or (not (unresolved-capacity?
                                      id value (get-in buffer-specs [id :elements])))
                                (and (= {:kind :plain} (:representation value))
                                     (nil? (:logical-layout value))))))
                        (:inputs operation)))
              (let [fold-reads (map-reads/fold-read-requirements operation map-read-options)
                    pure-map? (and (segop/seg-map? operation)
                                   (:out-sym operation)
                                   (not (seq (get-in operation [:scalar-region :effects]))))
                    independent-reads (when pure-map?
                                  ;; Read footprints do not authorize in-place execution.
                                  ;; The map owner separately checks same-lane alias legality.
                                  (map-reads/independent-read-requirements
                                   operation map-read-options))
                    optional-reads (merge-with
                                    (fn [left right]
                                      (if (= left right) left (launch/maximum left right)))
                                    (:requirements certificate) independent-reads fold-reads
                                    (get transform-read-requirements (:algorithm-equation operation)))
                    static-reads (when (and pure-map?
                                            (some #(not (contains? optional-reads %))
                                                  (:inputs operation)))
                                   (map-reads/static-read-requirements operation map-read-options))
                    derived (merge-with (fn [left right]
                                          (if (= left right) left (launch/maximum left right)))
                                        optional-reads static-reads)
                    _ (doseq [id marked-inputs
                              :when (not (contains? derived id))]
                          (fail! :scheduled-equation-read-capacity
                                 "pointwise storage refinement must retain its read-span proof"
                                 {:operation (:id operation) :phase (:phase operation)
                                  :input id :source operation}))]
                (merge-with into requirements
                            (into {} (map (fn [[id extent]] [id [extent]])) derived)))
              requirements)))
          (merge-with into
                      (contraction-read-requirements algorithm retained-equations)
                      (product-read-requirements values operations derived-scalars)
                      (reduce (fn [requirements reads]
                                (merge-with into requirements
                                            (into {} (map (fn [[id minimum]] [id [minimum]])) reads)))
                              {} (vals transform-read-requirements)))
          (map vector operations read-capacity-certificates))
         buffer-specs (reduce-kv
                       (fn [specs id requirements]
                         (if (not (unresolved-capacity?
                                    id (get values id) (get-in specs [id :elements])))
                           specs
                           (let [extents (vec (distinct requirements))
                                 required (if (= 1 (count extents)) (first extents)
                                              (apply launch/maximum extents))]
                             (assoc-in specs [id :elements] required))))
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
         (->> (concat (mapcat (comp seq :requirements) read-capacity-certificates)
                      (mapcat (fn [[id requirements]] (map #(vector id %) requirements))
                              read-requirements)
                      ;; A declared result shape is allocation capacity, not proof that
                      ;; it covers the canonical reduction's dense written prefix.
                      (mapcat (fn [equation]
                                (when (= 'segmented-reduce (soac/operation-kind equation))
                                  (try
                                    (seq (typed-projection/segmented-reduce-core-write-requirements
                                          algorithm equation))
                                    (catch clojure.lang.ExceptionInfo error
                                      (if (= :typed-soac-contract-projection
                                             (:reason (ex-data error)))
                                        nil
                                        (throw error))))))
                              retained-equations))
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

(defn- validated-scheduled-plan-operation?
  [plan operation]
  (cond
    (= plan operation) true
    (scheduled-body/scheduled-kernel-body? operation)
    (try
      (= plan (:source (scheduled-body/validate! operation)))
      (catch clojure.lang.ExceptionInfo _ false))
    :else false))

(defn- plan-equation-boundary?
  [values equation algorithm]
  (cond
    (swr/plan? algorithm)
    (and (swr/equation-boundary? values equation algorithm)
         (= 1 (count (:operations equation)))
         (validated-scheduled-plan-operation? algorithm (first (:operations equation))))

    (soac/program-form? algorithm)
    (algorithm-boundary? equation algorithm)

    :else false))

(defn- plan-output-allocations!
  [parallel-program plan]
  (let [{:keys [id dtype elements]} (:output plan)
        allocations (filterv #(= id (:destination %))
                             (get-in parallel-program [:attributes :allocations] []))
        fields #{:destination :source-binding-id :extent :initialization :dtype}
        valid? (fn [allocation]
                 (and (contains? #{fields (conj fields :source-expression)}
                                 (set (keys allocation)))
                      (or (not (contains? allocation :source-expression))
                          (seq? (:source-expression allocation)))
                      (integer? (:source-binding-id allocation))
                      (not (neg? (:source-binding-id allocation)))
                      (or (= elements (:extent allocation))
                          (and (symbol? (:extent allocation))
                               (integral-scalar-value?
                                (get-in parallel-program [:values (:extent allocation)]))))
                      (= dtype (:dtype allocation))
                      (contains? #{:zero :copy :unspecified}
                                 (:initialization allocation))))]
    (when-not (and (<= (count allocations) 1) (every? valid? allocations))
      (fail! :segmented-plan-output-allocation
             "segmented plan output has a malformed or conflicting allocation contract"
             {:output id :descriptor (:output plan) :allocations allocations}))
    ;; Keep the exact contract. In particular, :zero may not be discarded merely because the
    ;; semantic graph has a :write use; only a later body-specific complete-write witness can
    ;; authorize that optimization. Unrelated allocations remain owned by their equations.
    allocations))

(defn- descriptor-buffer
  [values descriptor role]
  (let [id (:id descriptor)
        value (get values id)]
    (graph/buffer id (:dtype descriptor) (swr/descriptor-launch-elements descriptor)
                  (or (:memory-space value) :device) role)))

(defn- plan-public-scalars
  [values plan]
  (mapv (fn [id]
          ;; equation-boundary? already proves scalar shape, integral dtype and plain storage.
          ;; Preserve that retained width: int-valued leaves remain int and may be declined by a
          ;; schedule whose private body only supports long.
          (graph/scalar id (get-in values [id :dtype])))
        (ordered-distinct (filter symbol? (swr/runtime-parameter-values plan)))))

(defn make-for-plan-equation
  "Derive the schedule-neutral semantic graph for one exact segmented-reduction equation.

   The returned body retains the equation's preceding host-scalar proof terms and either its typed
   plan operation or an exact ScheduledKernelBody refinement. The graph deliberately stores the
   original plan as its sole operation; target scheduling and emission are later phases. An exact
   output-allocation contract is retained, never elided from a mere graph write permission."
  [parallel-program equation]
  (let [parallel-program (program/validate! parallel-program)
        plan (:algorithm equation)
        _ (when-not (and (swr/plan? plan)
                         (plan-equation-boundary? (:values parallel-program) equation plan))
            (fail! :segmented-plan-equation-boundary
                   "segmented graph construction requires one exact validated plan equation"
                   {:equation (:id equation) :algorithm plan}))
        output-allocations (plan-output-allocations! parallel-program plan)
        body (dependency-closed-body
              parallel-program [equation]
              {:dialect (:dialect parallel-program)
               :source-dialect (:dialect parallel-program)
               :operation? #(or (swr/plan? %)
                                (scheduled-body/scheduled-kernel-body? %))
               :algorithm? #(plan-equation-boundary? (:values parallel-program) %1 %2)
               :body-attributes (when (seq output-allocations)
                                  {:allocations output-allocations})})
        input-buffers (mapv #(descriptor-buffer (:values body) % :input) (:operands plan))
        output-buffer (descriptor-buffer (:values body) (:output plan) :output)
        plan-scalars (plan-public-scalars (:values body) plan)
        scalar-ids (set (map :id plan-scalars))
        allocation-extent (:extent (first output-allocations))
        allocation-check? (and allocation-extent
                               (not= allocation-extent (get-in plan [:output :elements])))
        available (set (concat (:inputs body)
                               (mapcat :results (filter #(get-in % [:attributes :host-only])
                                                        (:equations body)))))
        _ (when (and allocation-check? (not (contains? available allocation-extent)))
            (fail! :segmented-plan-allocation-scope
                   "output allocation guard requires an earlier typed host scalar"
                   {:extent allocation-extent :available available}))
        scalars (cond-> plan-scalars
                  (and allocation-check? (not (contains? scalar-ids allocation-extent)))
                  (conj (graph/scalar allocation-extent
                                      (get-in body [:values allocation-extent :dtype]))))
        uses (into (mapv #(graph/->ValueUse (:id %) :read) input-buffers)
                   [(graph/->ValueUse (:id output-buffer) :write)])
        node (graph/->ScheduledKernel
              [:segmented-weighted-reduction (:id plan)] plan uses scalar-ids [])
        kernel-graph
        (graph/make
         {:inputs input-buffers
          :outputs [output-buffer]
          :scalars scalars
          :preconditions (if allocation-check?
                           [{:expression allocation-extent :op :=
                             :value (swr/descriptor-launch-elements (:output plan))}]
                           [])
          :nodes [node]
          :effects {:semantic (:effects equation)}
          :provenance {:source-dialect :segmented-weighted-reduction
                       :algorithm-dialect :segmented-weighted-reduction
                       :schedule-dialect :semantic-plan
                       :pass :scheduled-equation-graph}
          :attributes {:equation (:id equation)
                       :plan-id (:id plan)
                       :full-result-descriptor (:output plan)
                       :output-allocation (first output-allocations)}})]
    {:body body :graph kernel-graph}))
