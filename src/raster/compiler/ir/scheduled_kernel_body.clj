(ns raster.compiler.ir.scheduled-kernel-body
  "A checked semantic-operation to target-neutral KernelBody refinement.

   KernelBody fixes execution and memory structure.  ScheduledKernelBody binds that body to the
   exact operation it implements, its ordered compiler arguments, canonical memory effects, a
   named legality witness, and one shared numerical contract.  Target emitters consume this value;
   they must not reconstruct any of these facts from operation names or generated source."
  (:require [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.numerical-contract :as numerics]))

(defrecord ScheduledKernelBody
           [source body arguments scalar-bindings effects legality numerics provenance attributes])

(defn scheduled-kernel-body?
  [value]
  (and value
       (= "raster.compiler.ir.scheduled_kernel_body.ScheduledKernelBody"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :scheduled-kernel-body))))

(defn- record-kind? [simple-name value]
  (= (str "raster.compiler.ir.kernel_body." simple-name)
     (some-> value class .getName)))

(defn- nested-operations [operation]
  (cond
    (record-kind? "IfRegion" operation)
    (concat (:then-operations operation) (:else-operations operation))
    (or (record-kind? "ForLoop" operation) (record-kind? "PipelinedFor" operation)
        (record-kind? "Loop" operation) (record-kind? "Guard" operation))
    (:operations operation)
    :else []))

(defn- body-operations [kernel-body]
  (letfn [(walk [operations]
            (mapcat #(cons % (walk (nested-operations %))) operations))]
    (walk (:operations kernel-body))))

(defn- parameter-access
  [kind]
  (case kind
    :input :read
    :output :write
    :inout :read-write
    :scalar nil))

(defn derive-uses
  "Derive the canonical ordered external memory uses from body parameters and arguments.

   Reusing a compiler value in several read-only positions is harmless and collapses to one use.
   A repeated binding involving a write is not an alias proof and is rejected here. In-place
   access is represented by one `:inout` parameter, not split input/output parameters."
  [kernel-body arguments]
  (let [kernel-body (body/validate! kernel-body)]
    (when-not (and (vector? arguments) (= (count (:parameters kernel-body)) (count arguments)))
      (fail! :scheduled-kernel-body-arguments
             "scheduled body arguments must align one-to-one with KernelBody parameters"
             {:parameters (mapv :id (:parameters kernel-body)) :arguments arguments}))
    (reduce
     (fn [uses [parameter argument]]
       (if-let [access (parameter-access (:kind parameter))]
         (if-let [prior (some #(when (= argument (:value %)) %) uses)]
           (if (and (= :read (:access prior)) (= :read access))
             uses
             (fail! :scheduled-kernel-body-alias
                    "one storage value is bound to several parameters involving a write"
                    {:argument argument :prior prior :parameter parameter :access access}))
           (conj uses {:value argument :access access}))
         uses))
     [] (map vector (:parameters kernel-body) arguments))))

(defn derive-scalar-bindings
  "Derive ordered identity scalar bindings. Schedules needing a representation conversion must
   construct and certify that conversion explicitly rather than relying on an ABI rewrite."
  [kernel-body arguments]
  (let [kernel-body (body/validate! kernel-body)]
    (mapv (fn [[parameter argument]]
            {:parameter (:id parameter)
             :value argument
             :dtype (:dtype parameter)
             :kernel-dtype (:dtype parameter)
             :conversion :identity})
          (filter (fn [[parameter _]] (= :scalar (:kind parameter)))
                  (map vector (:parameters kernel-body) arguments)))))

(defn- validate-scalar-bindings!
  [kernel-body arguments scalar-bindings]
  (let [expected (derive-scalar-bindings kernel-body arguments)
        valid-binding?
        (fn [identity-binding actual]
          (let [logical-dtype (some-> (:dtype actual) dtype/canon)
                kernel-dtype (some-> (:kernel-dtype actual) dtype/canon)
                physical-dtype (:dtype identity-binding)
                same-identity? (= (select-keys identity-binding [:parameter :value])
                                  (select-keys actual [:parameter :value]))]
            (and same-identity?
                 (= physical-dtype kernel-dtype)
                 (= (:dtype actual) logical-dtype)
                 (= (:kernel-dtype actual) kernel-dtype)
                 (case (:conversion actual)
                   :identity (= logical-dtype kernel-dtype)
                   ;; A logical graph extent may deliberately use a wider host representation
                   ;; than a target-private ABI slot. The graph-call boundary range-checks this
                   ;; conversion before driver contact; schedules must state the proof rather
                   ;; than obtaining a narrowing from target emission by accident.
                   :checked-range (and (= :long logical-dtype) (= :int kernel-dtype))
                   false))))]
    (when-not (and (vector? scalar-bindings)
                   (= (count expected) (count scalar-bindings))
                   (every? true? (map valid-binding? expected scalar-bindings)))
      (fail! :scheduled-kernel-body-scalar-bindings
             "scheduled scalar bindings must explicitly preserve or prove logical-to-kernel dtype conversion"
             {:expected expected :actual scalar-bindings}))
    scalar-bindings))

(defn- launch-expressions
  [launch-spec]
  (concat (:workgroup-size launch-spec) (:group-count launch-spec)))

(defn realized-launch
  "Return the checked semantic launch after substituting ordered compiler arguments."
  [scheduled]
  (let [{kernel-body :body arguments :arguments} scheduled
        kernel-body (body/validate! kernel-body)
        parameters (:parameters kernel-body)
        substitutions (into {} (map (fn [[parameter argument]] [(:id parameter) argument]))
                            (map vector parameters arguments))
        scalar-parameters (filterv #(= :scalar (:kind %)) parameters)
        parameter-types (into {} (map (juxt :id :dtype)) scalar-parameters)
        allowed-parameters (set (keys parameter-types))
        body-launch (:launch kernel-body)]
    (doseq [expression (launch-expressions body-launch)
            reference (launch/expression-references expression)]
      (when-not (contains? allowed-parameters reference)
        (fail! :scheduled-kernel-body-launch-closure
               "KernelBody launch references a non-scalar or absent parameter"
               {:reference reference :allowed allowed-parameters :launch body-launch})))
    (doseq [expression (launch-expressions body-launch)]
      (launch/validate-typed-expression! expression parameter-types))
    ;; Compiler arguments may themselves be checked graph-derived integer expressions. Their
    ;; leaves and logical dtypes belong to the surrounding KernelGraph, so standalone validation
    ;; proves structural substitution while `validate-against-node!` closes that context.
    (launch/rebind-spec body-launch substitutions)))

(defn validate!
  [scheduled]
  (when-not (scheduled-kernel-body? scheduled)
    (fail! :scheduled-kernel-body-type "expected a ScheduledKernelBody"
           {:actual (type scheduled)}))
  (let [{:keys [source arguments scalar-bindings effects legality numerics provenance attributes]
         kernel-body :body} scheduled
        kernel-body (body/validate! kernel-body)
        expected-uses (derive-uses kernel-body arguments)]
    (when (nil? source)
      (fail! :scheduled-kernel-body-source
             "scheduled kernel body requires its exact source operation" {}))
    (when (some nil? arguments)
      (fail! :scheduled-kernel-body-arguments
             "scheduled body arguments cannot contain nil compiler values"
             {:arguments arguments}))
    (validate-scalar-bindings! kernel-body arguments scalar-bindings)
    (when-not (and (map? effects) (keyword? (:kind effects)))
      (fail! :scheduled-kernel-body-effects
             "scheduled body requires a named canonical effects map"
             {:actual effects}))
    (when-not (= expected-uses (:uses effects))
      (fail! :scheduled-kernel-body-effects
             "scheduled body memory effects differ from its ordered parameter binding"
             {:expected expected-uses :actual (:uses effects)}))
    (let [expected-reads (mapv :value (filter #(contains? #{:read :read-write} (:access %))
                                              expected-uses))
          expected-writes (mapv :value (filter #(contains? #{:write :read-write} (:access %))
                                               expected-uses))]
      (doseq [[field expected] [[:reads expected-reads] [:writes expected-writes]]
              :when (contains? effects field)]
        (when-not (= expected (get effects field))
          (fail! :scheduled-kernel-body-effects
                 "scheduled body carries a contradictory derived effect projection"
                 {:field field :expected expected :actual (get effects field)}))))
    (when-not (and (map? legality) (keyword? (:kind legality)))
      (fail! :scheduled-kernel-body-legality
             "scheduled kernel body requires a named legality witness"
             {:legality legality}))
    (numerics/validate! numerics {:reason :scheduled-kernel-body-numerics
                                  :ir :scheduled-kernel-body})
    (realized-launch scheduled)
    (doseq [[field value] [[:legality legality] [:numerics numerics]
                           [:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :scheduled-kernel-body-description
               "scheduled body descriptions must be maps"
               {:field field :value value})))
    scheduled))

(defn make
  [{:keys [source body arguments scalar-bindings effects legality numerics provenance attributes]
    :or {provenance {} attributes {}}}]
  (let [scalar-bindings (or scalar-bindings (derive-scalar-bindings body arguments))]
    (validate! (->ScheduledKernelBody source body arguments scalar-bindings effects legality
                                      numerics provenance attributes))))

(defn validate-against-node!
  "Require this refinement to implement one exact node in its complete KernelGraph context."
  [scheduled node kernel-graph]
  (when-not (graph/scheduled-kernel? node)
    (fail! :scheduled-kernel-body-node
           "scheduled body graph context must be a ScheduledKernel"
           {:actual (type node)}))
  (let [scheduled (validate! scheduled)
        kernel-graph (graph/validate! kernel-graph)
        expected (into {} (map (juxt :buffer :access)) (:uses node))
        actual (into {} (map (juxt :value :access)) (get-in scheduled [:effects :uses]))
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars kernel-graph))
        scalar-arguments (mapv :value (:scalar-bindings scheduled))
        actual-scalars (reduce into #{} (map launch/expression-references scalar-arguments))
        expected-scalars (set (:scalar-uses node))]
    (when (nil? (:scalar-uses node))
      (fail! :scheduled-kernel-body-node-scalars
             "scheduled-body certification requires explicit graph-node scalar dependencies"
             {:node (:id node)}))
    (when-not (= (:source scheduled) (:operation node))
      (fail! :scheduled-kernel-body-source
             "scheduled body does not refine the graph node's exact operation"
             {:node (:id node)}))
    (when-not (= expected actual)
      (fail! :scheduled-kernel-body-node-effects
             "scheduled body pointer effects differ from the graph node uses"
             {:node (:id node) :expected expected :actual actual}))
    (when-not (= expected-scalars actual-scalars)
      (fail! :scheduled-kernel-body-node-scalars
             "scheduled body scalar dependencies differ from its source operation"
             {:node (:id node) :expected expected-scalars :actual actual-scalars}))
    (doseq [{:keys [value dtype]} (:scalar-bindings scheduled)]
      (if (contains? #{:int :long} dtype)
        (let [actual-dtype (launch/typed-expression-dtype value scalar-types)]
          (when-not (= dtype actual-dtype)
            (fail! :scheduled-kernel-body-node-scalar-dtype
                   "scheduled integer binding requires an explicit representation conversion"
                   {:node (:id node) :value value :expected dtype :actual actual-dtype})))
        (when-not (= dtype (get scalar-types value))
          (fail! :scheduled-kernel-body-node-scalar-dtype
                 "scheduled scalar binding differs from its GraphScalar dtype"
                 {:node (:id node) :value value :expected dtype
                  :actual (get scalar-types value)}))))
    scheduled))

(defn validate-artifact-projection!
  "Validate the structural projection of this exact ScheduledKernelBody into an artifact.

   Target emitters remain the trusted lowering boundary for source semantics; this check proves
   that their artifact did not change the scheduled arguments, effects, launch, body, ABI memory
   preconditions, or identity-bearing physical facts."
  [scheduled emitted]
  (let [scheduled (validate! scheduled)
        emitted (artifact/validate! emitted)
        parameters (get-in scheduled [:body :parameters])
        bindings (into {} (map (juxt :parameter identity)) (:scalar-bindings scheduled))
        target-facts (get-in emitted [:attributes :target-facts])
        pointer-alignment (:pointer-alignment target-facts)
        expected-slots
        (body-abi/project-contracts
         (mapv (fn [index {:keys [id kind dtype role]}]
                 (let [binding (get bindings id)]
                   (abi/slot id kind (or (:dtype binding) dtype)
                             :kernel-dtype (or (:kernel-dtype binding) dtype)
                             :c-name (str "p" index) :role role
                             :alignment (when (and pointer-alignment (not= :scalar kind))
                                          pointer-alignment))))
               (range) parameters)
         (:body scheduled))
        fields [:kind :dtype :kernel-dtype :role :aliasing :alignment]
        expected-abi (mapv #(select-keys % fields) expected-slots)
        actual-abi (mapv #(select-keys % fields) (:abi emitted))
        matrix-instructions (->> (body-operations (:body scheduled))
                                 (filter #(record-kind? "MatrixMad" %))
                                 (map :instruction) distinct vec)
        expected-target ({:opencl-intel :opencl-c :opencl-portable :opencl-c
                          :cuda :cuda-c :hip :hip-cpp}
                         (:target-dialect target-facts))]
    (doseq [[field expected actual]
            [[:certificate scheduled (get-in emitted [:provenance :scheduled-operation])]
             [:arguments (:arguments scheduled) (:arguments emitted)]
             [:effects (:effects scheduled) (:effects emitted)]
             [:launch (realized-launch scheduled) (:launch emitted)]
             [:body (:body scheduled) (get-in emitted [:attributes :kernel-body])]
             [:abi expected-abi actual-abi]]]
      (when-not (= expected actual)
        (fail! :scheduled-kernel-body-artifact-projection
               "target artifact differs from its embedded scheduled-body certificate"
               {:field field :expected expected :actual actual})))
    (when-not (= target-facts (get-in emitted [:provenance :target-facts]))
      (fail! :scheduled-kernel-body-artifact-projection
             "artifact target facts disagree across provenance and attributes"
             {:attributes target-facts
              :provenance (get-in emitted [:provenance :target-facts])}))
    (when-not (= expected-target (:target emitted))
      (fail! :scheduled-kernel-body-artifact-projection
             "artifact target module disagrees with its target dialect fact"
             {:target-dialect (:target-dialect target-facts)
              :expected expected-target :actual (:target emitted)}))
    (when (seq matrix-instructions)
      (when-not (and (= 1 (count matrix-instructions))
                     (= (first matrix-instructions) (:instruction target-facts))
                     (= (get-in (first matrix-instructions) [:family])
                        (:instruction-family target-facts)))
        (fail! :scheduled-kernel-body-artifact-projection
               "matrix target facts differ from the scheduled MatrixMad instruction"
               {:instructions matrix-instructions :target-facts target-facts}))
      (when (and (= :cuda (:target-dialect target-facts))
                 (= :mma (:instruction-family target-facts))
                 (not= 32 pointer-alignment))
        (fail! :scheduled-kernel-body-artifact-projection
               "CUDA MMA projection requires its verified pointer alignment"
               {:required 32 :actual pointer-alignment})))
    emitted))
