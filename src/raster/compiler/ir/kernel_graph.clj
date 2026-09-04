(ns raster.compiler.ir.kernel-graph
  "Verified scheduling IR for algorithms that require one or more kernels.

   KernelGraph is deliberately earlier than target emission: a ScheduledKernel owns a SegOp (or a
   future target-neutral kernel body), explicit buffer uses, and dependencies.  Emission later
   replaces the operation with a KernelArtifact without reconstructing dataflow from marker
   conventions.  Stable GraphBuffer identities make intermediate storage part of the program."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.segop :as segop]))

(defrecord GraphBuffer [id dtype elements memory-space role])
(defrecord GraphScalar [id dtype])
(defrecord ValueUse [buffer access])
(defrecord ScheduledKernel [id operation uses dependencies])
(defrecord KernelGraph
           [inputs outputs temporaries scalars nodes abi arguments effects provenance attributes])

(def ^:private buffer-roles #{:input :output :inout :temporary})
(def ^:private access-modes #{:read :write :read-write})

(defn- record-kind? [record-class value]
  (and value (= record-class (.getName (class value)))))

(defn graph-buffer? [x]
  (record-kind? "raster.compiler.ir.kernel_graph.GraphBuffer" x))

(defn graph-scalar? [x]
  (record-kind? "raster.compiler.ir.kernel_graph.GraphScalar" x))

(defn value-use? [x]
  (record-kind? "raster.compiler.ir.kernel_graph.ValueUse" x))

(defn scheduled-kernel? [x]
  (record-kind? "raster.compiler.ir.kernel_graph.ScheduledKernel" x))

(defn kernel-graph? [x]
  (record-kind? "raster.compiler.ir.kernel_graph.KernelGraph" x))

(defn- validate-buffer-fields! [{:keys [id dtype elements memory-space role] :as value}]
  (when (nil? id)
    (throw (ex-info "graph buffer requires an identity" {:id id})))
  (when-not (keyword? dtype)
    (throw (ex-info "graph buffer requires a dtype" {:id id :dtype dtype})))
  (when-not (contains? buffer-roles role)
    (throw (ex-info "graph buffer has an invalid role" {:id id :role role})))
  (when-not (keyword? memory-space)
    (throw (ex-info "graph buffer requires a memory space" {:id id :memory-space memory-space})))
  (when (and (= :temporary role) (nil? elements))
    (throw (ex-info "temporary graph buffer requires an element count" {:id id})))
  value)

(defn buffer
  "Construct a scheduled buffer. `elements` may be symbolic, but a temporary must state it."
  [id dtype elements memory-space role]
  (validate-buffer-fields! (->GraphBuffer id dtype elements memory-space role)))

(defn scalar
  "Construct one ordered, target-neutral public scalar dependency."
  [id scalar-dtype]
  (when-not (or (symbol? id) (keyword? id))
    (throw (ex-info "graph scalar identity must be a symbol or keyword" {:id id})))
  (when-not (dtype/known? scalar-dtype)
    (throw (ex-info "graph scalar requires a known dtype"
                    {:id id :dtype scalar-dtype})))
  (->GraphScalar id (dtype/canon scalar-dtype)))

(defn- reads? [access] (contains? #{:read :read-write} access))
(defn- writes? [access] (contains? #{:write :read-write} access))

(defn- access-sets [node]
  (reduce (fn [{:keys [reads writes] :as result} use]
            (cond-> result
              (reads? (:access use)) (assoc :reads (conj reads (:buffer use)))
              (writes? (:access use)) (assoc :writes (conj writes (:buffer use)))))
          {:reads #{} :writes #{}}
          (:uses node)))

(defn- hazard? [earlier later]
  (let [{er :reads ew :writes} (access-sets earlier)
        {lr :reads lw :writes} (access-sets later)]
    (or (seq (set/intersection ew lr))
        (seq (set/intersection ew lw))
        (seq (set/intersection er lw)))))

(defn has-interface?
  "Return true when a graph declares an ordered external call interface. Target-neutral graphs
   may omit it temporarily; every emitted executable graph must provide it."
  [graph]
  (and (some? (:abi graph)) (some? (:arguments graph))))

(defn- validate-interface!
  [graph]
  (let [{:keys [abi arguments scalars]} graph]
    (when-not (= (some? abi) (some? arguments))
      (throw (ex-info "kernel graph ABI and arguments must be declared together"
                      {:abi abi :arguments arguments})))
    (when (some? abi)
      (kabi/validate-arguments! abi arguments)
      (let [external-by-id (into {} (map (juxt :id identity))
                                 (concat (:inputs graph) (:outputs graph)))
            pointer-pairs (filterv (fn [[slot _]] (not= :scalar (:kind slot)))
                                   (mapv vector abi arguments))
            pointer-arguments (mapv second pointer-pairs)
            scalar-pairs (filterv (fn [[slot _]] (= :scalar (:kind slot)))
                                  (mapv vector abi arguments))
            scalar-arguments (mapv second scalar-pairs)]
        (when-not (= (set (keys external-by-id)) (set pointer-arguments))
          (throw (ex-info "kernel graph pointer interface differs from its external buffers"
                          {:external (set (keys external-by-id))
                           :pointer-arguments pointer-arguments})))
        (when-not (= (count pointer-arguments) (count (set pointer-arguments)))
          (throw (ex-info "kernel graph pointer interface must name each external buffer once"
                          {:pointer-arguments pointer-arguments})))
        (when-not (= (count scalar-arguments) (count (set scalar-arguments)))
          (throw (ex-info "kernel graph scalar interface arguments must be unique"
                          {:scalar-arguments scalar-arguments})))
        (when (some? scalars)
          (let [expected (mapv (juxt :id :dtype) scalars)
                actual (mapv (fn [[slot argument]]
                               [argument (dtype/canon (:dtype slot))])
                             scalar-pairs)]
            (when-not (= expected actual)
              (throw (ex-info "kernel graph ABI scalar interface differs from its schedule"
                              {:reason :kernel-graph-scalar-interface
                               :expected expected :actual actual})))))
        (doseq [[slot id] pointer-pairs]
          (let [{:keys [dtype role]} (get external-by-id id)
                expected-kind (case role
                                :input :input
                                :output :output
                                :inout :inout)]
            (when-not (= dtype (:dtype slot))
              (throw (ex-info "kernel graph ABI storage dtype differs from its buffer"
                              {:buffer id :buffer-dtype dtype :slot slot})))
            (when-not (= expected-kind (:kind slot))
              (throw (ex-info "kernel graph ABI pointer direction differs from its buffer role"
                              {:buffer id :role role :slot slot
                               :expected-kind expected-kind})))))))
    graph))

(defn validate!
  "Validate a KernelGraph and return it unchanged.

   Besides structural checks this proves the property that motivated the first graph conversion:
   every temporary is written before it is read, and every memory hazard is represented by a
   dependency."
  [graph]
  (when-not (kernel-graph? graph)
    (throw (ex-info "kernel graph must be a KernelGraph value"
                    {:graph graph :actual (type graph)})))
  (let [{:keys [inputs outputs temporaries scalars nodes effects provenance attributes]} graph
        sections [inputs outputs temporaries]
        buffers (vec (mapcat identity sections))
        buffer-ids (mapv :id buffers)]
    (doseq [[field values] [[:inputs inputs] [:outputs outputs] [:temporaries temporaries]
                            [:nodes nodes]]]
      (when-not (vector? values)
        (throw (ex-info "kernel graph sections must be ordered vectors"
                        {:field field :value values}))))
    (when-not (or (nil? scalars) (vector? scalars))
      (throw (ex-info "kernel graph scalar interface must be an ordered vector"
                      {:field :scalars :value scalars})))
    (when (some? scalars)
      (doseq [value scalars]
        (when-not (graph-scalar? value)
          (throw (ex-info "kernel graph contains a non-scalar interface value"
                          {:scalar value})))
        (when-not (or (symbol? (:id value)) (keyword? (:id value)))
          (throw (ex-info "graph scalar identity must be a symbol or keyword"
                          {:scalar value})))
        (when-not (and (dtype/known? (:dtype value))
                       (= (:dtype value) (dtype/canon (:dtype value))))
          (throw (ex-info "kernel graph scalar must use its canonical dtype"
                          {:scalar value}))))
      (when-not (= (count scalars) (count (distinct (map :id scalars))))
        (throw (ex-info "kernel graph scalar identities must be unique"
                        {:scalars scalars})))
      (let [collisions (set/intersection (set buffer-ids) (set (map :id scalars)))]
        (when (seq collisions)
          (throw (ex-info "kernel graph buffer and scalar identities must be disjoint"
                          {:reason :kernel-graph-value-identity
                           :collisions collisions})))))
    (doseq [b buffers]
      (when-not (graph-buffer? b)
        (throw (ex-info "kernel graph contains a non-buffer value" {:buffer b})))
      (validate-buffer-fields! b))
    (doseq [[field values allowed-roles]
            [[:inputs inputs #{:input :inout}]
             [:outputs outputs #{:output :inout}]
             [:temporaries temporaries #{:temporary}]]
            b values]
      (when-not (contains? allowed-roles (:role b))
        (throw (ex-info "kernel graph buffer role disagrees with its section"
                        {:field field :buffer b :allowed-roles allowed-roles}))))
    ;; An in-place value intentionally occurs in both external sections. It must be the same
    ;; GraphBuffer value, not two descriptions that happen to reuse a symbol.
    (doseq [[id occurrences] (group-by :id buffers)]
      (when (or (> (count occurrences) 2)
                (and (= 2 (count occurrences))
                     (not (and (apply = occurrences)
                               (= :inout (:role (first occurrences)))
                               (some #(= id (:id %)) inputs)
                               (some #(= id (:id %)) outputs)))))
        (throw (ex-info "kernel graph buffer identity is declared inconsistently"
                        {:id id :occurrences occurrences}))))
    (doseq [[field value] [[:effects effects] [:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (throw (ex-info "kernel graph descriptive sections must be maps"
                        {:field field :value value}))))
    (validate-interface! graph)
    (let [declared (set buffer-ids)
          output-ids (set (map :id outputs))
          initially-written (set (map :id (filter #(contains? #{:input :inout} (:role %)) buffers)))]
      (loop [remaining nodes
             preceding []
             node-ids #{}
             initialized initially-written
             graph-writes #{}]
        (if-let [node (first remaining)]
          (do
            (when-not (scheduled-kernel? node)
              (throw (ex-info "kernel graph contains a non-scheduled node" {:node node})))
            (when (nil? (:id node))
              (throw (ex-info "scheduled kernel requires an identity" {:node node})))
            (when (contains? node-ids (:id node))
              (throw (ex-info "scheduled kernel identities must be unique" {:id (:id node)})))
            (when-not (vector? (:uses node))
              (throw (ex-info "scheduled kernel uses must be an ordered vector" {:node (:id node)})))
            (when-not (vector? (:dependencies node))
              (throw (ex-info "scheduled kernel dependencies must be an ordered vector"
                              {:node (:id node)})))
            (when (nil? (:operation node))
              (throw (ex-info "scheduled kernel requires an operation" {:node (:id node)})))
            (let [uses (:uses node)
                  use-ids (mapv :buffer uses)
                  duplicate-use (not= (count use-ids) (count (set use-ids)))
                  dependency-set (set (:dependencies node))
                  preceding-ids (set (map :id preceding))]
              (doseq [use uses]
                (when-not (value-use? use)
                  (throw (ex-info "scheduled kernel contains a non-use value"
                                  {:node (:id node) :use use})))
                (when-not (contains? access-modes (:access use))
                  (throw (ex-info "scheduled kernel use has an invalid access mode"
                                  {:node (:id node) :use use})))
                (when-not (contains? declared (:buffer use))
                  (throw (ex-info "scheduled kernel uses an undeclared buffer"
                                  {:node (:id node) :buffer (:buffer use)}))))
              (when duplicate-use
                (throw (ex-info "scheduled kernel must combine access to each buffer into one use"
                                {:node (:id node) :uses use-ids})))
              (when-not (= (count dependency-set) (count (:dependencies node)))
                (throw (ex-info "scheduled kernel dependencies must be unique"
                                {:node (:id node) :dependencies (:dependencies node)})))
              (when-not (set/subset? dependency-set preceding-ids)
                (throw (ex-info "scheduled kernel dependency must name an earlier node"
                                {:node (:id node) :dependencies (:dependencies node)
                                 :earlier preceding-ids})))
              (let [missing-hazards (->> preceding
                                         (filter #(hazard? % node))
                                         (map :id)
                                         (remove dependency-set)
                                         vec)
                    reads (->> uses (filter #(reads? (:access %))) (map :buffer) set)
                    writes (->> uses (filter #(writes? (:access %))) (map :buffer) set)
                    uninitialized (set/difference reads initialized)]
                (when (seq missing-hazards)
                  (throw (ex-info "scheduled kernel omits a memory-hazard dependency"
                                  {:node (:id node) :missing missing-hazards})))
                (when (seq uninitialized)
                  (throw (ex-info "scheduled kernel reads a buffer before any graph node writes it"
                                  {:node (:id node) :buffers uninitialized})))
                (recur (next remaining)
                       (conj preceding node)
                       (conj node-ids (:id node))
                       (set/union initialized writes)
                       (set/union graph-writes writes)))))
          (when-not (set/subset? output-ids graph-writes)
            (throw (ex-info "kernel graph does not write every output"
                            {:outputs output-ids :written graph-writes}))))))
    graph))

(defn make
  "Construct and verify a scheduled KernelGraph from explicit compiler values."
  [{:keys [inputs outputs temporaries scalars nodes abi arguments effects provenance attributes]
    :or {inputs [] outputs [] temporaries [] nodes [] effects {} provenance {} attributes {}}}]
  (validate! (->KernelGraph inputs outputs temporaries scalars nodes abi arguments
                            effects provenance attributes)))

(defn map-operations
  "Replace each scheduled operation while preserving and revalidating graph dataflow. `f` receives
   the complete ScheduledKernel, so target lowering can use phase-local uses and provenance."
  [graph f]
  (let [graph (validate! graph)]
    (validate! (update graph :nodes
                       #(mapv (fn [node] (assoc node :operation (f node))) %)))))

(defn dataflow-contract
  "Project the target-independent buffers, accesses, and ordering of a KernelGraph.

   Operations and the emitted external ABI are deliberately excluded: target lowering replaces
   the former and constructs the latter. Buffer contracts, node identities, uses, dependencies,
   and semantic graph effects may not change during that replacement."
  [graph]
  (let [graph (validate! graph)]
    {:inputs (:inputs graph)
     :outputs (:outputs graph)
     :temporaries (:temporaries graph)
     :scalars (:scalars graph)
     :nodes (mapv #(select-keys % [:id :uses :dependencies]) (:nodes graph))
     :effects (:effects graph)}))

(defn boundary-contract
  "Project the exact public boundary of a scheduled graph.

   A schedule refinement may introduce private buffers and replace one operation with several
   scheduled stages, but it may not change the ordered external storage contract, public ABI, or
   logical effects.  Keeping this projection separate from `dataflow-contract` makes that
   distinction explicit instead of weakening target-emission equivalence."
  [graph]
  (let [graph (validate! graph)]
    {:inputs (:inputs graph)
     :outputs (:outputs graph)
     :scalars (:scalars graph)
     :abi (:abi graph)
     :arguments (:arguments graph)
     :effects (:effects graph)}))

(defn schedule-contract
  "Project the complete scheduled identity of a graph before target emission.

   Unlike `dataflow-contract`, this includes the exact ordered operations.  It is used to bind a
   graph-refinement witness to the semantic schedule it claims to refine; target emission still
   uses the stricter positional one-operation-to-one-artifact check."
  [graph]
  (let [graph (validate! graph)]
    {:boundary (boundary-contract graph)
     :dataflow (dataflow-contract graph)
     :operations (mapv :operation (:nodes graph))}))

(defn dataflow-equivalent?
  "Whether two valid graphs have exactly the same target-independent dataflow contract."
  [left right]
  (= (dataflow-contract left) (dataflow-contract right)))

(defn- ordered [xs]
  (vec (sort-by pr-str xs)))

(defn- operation-use [input-ids output-ids id]
  (->ValueUse id (cond
                   (and (contains? input-ids id) (contains? output-ids id)) :read-write
                   (contains? output-ids id) :write
                   :else :read)))

(defn from-segops
  "Build and verify a scheduled graph from an already selected SegOp sequence.

   `temporaries` is a map from stable buffer identity to at least `{:elements expr}`. External
   inputs/outputs are explicit; therefore a newly introduced intermediate can never hide as an
   undeclared symbol in a later kernel."
  [segops {:keys [inputs outputs temporaries scalars buffer-specs dtype memory-space effects provenance
                  attributes]
           :or {temporaries {} buffer-specs {} memory-space :device effects {} provenance {}
                attributes {}}}]
  (when-not (vector? segops)
    (throw (ex-info "scheduled SegOps must be an ordered vector" {:segops segops})))
  (doseq [operation segops]
    (when-not (segop/segop? operation)
      (throw (ex-info "scheduled SegOp graph contains a non-SegOp operation"
                      {:operation operation :actual (type operation)}))))
  (let [input-ids (set inputs)
        output-ids (set outputs)
        external-ids (set/union input-ids output-ids)
        operation-ids (reduce set/union #{}
                              (map #(set/union (set (or (segop/segop-inputs %) #{}))
                                               (set (or (segop/segop-outputs %) #{})))
                                   segops))
        temporary-ids (set (keys temporaries))
        expected-temporaries (set/difference operation-ids external-ids)]
    (when-not (= expected-temporaries temporary-ids)
      (throw (ex-info "scheduled temporary declarations differ from SegOp dataflow"
                      {:expected expected-temporaries :declared temporary-ids})))
    (let [external-buffers (into {}
                                 (map (fn [id]
                                        (let [spec (get buffer-specs id)]
                                          [id (buffer id (or (:dtype spec) dtype)
                                                      (:elements spec)
                                                      (or (:memory-space spec) memory-space)
                                                      (cond
                                                        (contains? (set/intersection input-ids
                                                                                     output-ids) id)
                                                        :inout
                                                        (contains? input-ids id) :input
                                                        :else :output))])))
                                 (ordered external-ids))
          inputs* (mapv external-buffers (ordered input-ids))
          outputs* (mapv external-buffers (ordered output-ids))
          temporaries* (mapv (fn [id]
                               (let [spec (get temporaries id)]
                                 (buffer id
                                         (or (:dtype spec) dtype)
                                         (:elements spec)
                                         (or (:memory-space spec) memory-space)
                                         :temporary)))
                             (ordered temporary-ids))
          nodes (loop [ops segops index 0 previous [] result []]
                  (if-let [op (first ops)]
                    (let [ins (set (or (segop/segop-inputs op) #{}))
                          outs (set (or (segop/segop-outputs op) #{}))
                          uses (mapv #(operation-use ins outs %)
                                     (ordered (set/union ins outs)))
                          ;; A scheduled-node identity is not a SegOp identity. The position makes
                          ;; repeated/colliding source ids safe when larger graphs are assembled.
                          candidate (->ScheduledKernel [:segop (:id op) index] op uses [])
                          dependencies (->> previous
                                            (filter #(hazard? % candidate))
                                            (mapv :id))
                          node (assoc candidate :dependencies dependencies)]
                      (recur (next ops) (inc index) (conj previous node) (conj result node)))
                    result))]
      (make {:inputs inputs*
             :outputs outputs*
             :temporaries temporaries*
             :scalars scalars
             :nodes nodes
             :effects effects
             :provenance provenance
             :attributes attributes}))))
