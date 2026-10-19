(ns raster.compiler.ir.invocation-link
  "Pure lowering from a specialized public invocation to an equation-first LinkPlan.

   MaterializedInvocation owns public argument/scalar semantics. EmittedParallelProgram owns the
   numerical program. This namespace realizes their shared storage boundary exactly once: caller
   arrays become initialized LinkNodes, physical equation results and loop carry rotations become
   owned nodes, and the emitted call enters LinkPlan as a ProgramLinkInstance. No driver object,
   kernel name, or legacy resident descriptor participates."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.write-coverage :as coverage]
            [raster.compiler.ir.structured-control :as control]))

(defrecord InvocationLinkCertificate
           [source-dialect target-dialect plan-id target instance-ids public-buffer-bindings
            public-buffer-roles compiler-buffer-bindings semantic-outputs outputs aliases
            driver-allocations effect-evidence])
(defrecord CertifiedInvocationLink [plan certificate])

(defn certificate? [value]
  (instance? InvocationLinkCertificate value))

(defn certified-link? [value]
  (instance? CertifiedInvocationLink value))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :invocation-link))))

(defn- typed-scalar?
  [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn- scalar-number
  [scalars value-id storage-id]
  (let [scalar (get scalars value-id)]
    (when-not (and (typed-scalar? scalar)
                   (contains? #{:int :long} (:type scalar))
                   (integer? (:value scalar)))
      (fail! :invocation-link-shape-scalar
             "program storage shape requires an integral typed scalar"
             {:storage-value storage-id :shape-value value-id
              :scalar scalar :available (set (keys scalars))}))
    (long (:value scalar))))

(defn- invocation-shape-scalars
  [materialized]
  (let [plan (:plan materialized)
        values (:values materialized)]
    (reduce (fn [environment {:keys [id symbol]}]
              (let [value (get values id)]
                (if (typed-scalar? value)
                  (assoc environment symbol value)
                  environment)))
            {}
            (concat (:parameters plan) (:steps plan)))))

(defn- concrete-shape
  [id abstract scalars buffers storage]
  (mapv
   (fn [dimension]
     (let [dimension (if (and (seq? dimension) (= 'value (first dimension))
                              (= 2 (count dimension)))
                       (second dimension)
                       dimension)
           resolved
           (cond
             (integer? dimension) (long dimension)
             (or (symbol? dimension) (vector? dimension) (keyword? dimension))
             (scalar-number scalars dimension id)
             (and (seq? dimension) (= 'extent (first dimension)) (= 2 (count dimension)))
             (let [source-id (get buffers (second dimension))
                   source (get storage source-id)]
               (when-not source
                 (fail! :invocation-link-shape-extent
                        "program extent projection names unavailable resident storage"
                        {:dimension dimension :buffers (set (keys buffers))}))
               (first (:shape source)))
             (seq? dimension)
             (launch/resolve-expression
              #(scalar-number scalars % id)
              (index-expression/project-dimension
               dimension #(get-in scalars [% :type]) fail!))
             :else
             (fail! :invocation-link-shape-expression
                    "program storage shape is not a canonical integer/value/extent expression"
                    {:value-id id :dimension dimension :shape (:shape abstract)}))]
       (when (neg? resolved)
         (fail! :invocation-link-shape-negative
                "program storage dimensions must be non-negative"
                {:dimension dimension :resolved resolved}))
       resolved))
   (:shape abstract)))

(defn- storage-id
  [invocation-id kind compiler-value]
  [::storage invocation-id kind compiler-value])

(defn- add-storage
  [state token compiler-value abstract shape source initialization]
  (let [entry {:id token :compiler-values #{compiler-value} :abstract (av/validate! abstract)
               :shape (vec shape) :source source :initialization initialization}]
    (if-let [prior (get-in state [:storage token])]
      (do
        (when-not (and (= (:shape prior) (:shape entry))
                       (av/storage-contract-compatible? (:abstract prior) abstract))
          (fail! :invocation-link-storage-alias
                 "one materialized storage identity has incompatible compiler contracts"
                 {:storage token :first prior :compiler-value compiler-value
                  :abstract abstract :shape shape}))
        (update-in state [:storage token :compiler-values] conj compiler-value))
      (assoc-in state [:storage token] entry))))

(defn- allocate-result
  [state invocation-id compiler-value abstract scalars kind]
  (if-let [token (get-in state [:buffers compiler-value])]
    [state token]
    (let [token (storage-id invocation-id kind compiler-value)
          shape (concrete-shape compiler-value abstract scalars
                                (:buffers state) (:storage state))]
      [(-> state
           (assoc-in [:buffers compiler-value] token)
           (add-storage token compiler-value abstract shape nil :unspecified))
       token])))

(defn- graph-buffer-access
  [graph id]
  (let [roles (into #{}
                    (keep (fn [buffer] (when (= id (:id buffer)) (:role buffer))))
                    (concat (:inputs graph) (:outputs graph)))]
    (cond
      (contains? roles :inout) :read-write
      (and (contains? roles :input) (contains? roles :output)) :read-write
      (contains? roles :input) :read
      (contains? roles :output) :write
      :else nil)))

(defn- soac-complete-write?
  [operation id capacity scalars buffers storage]
  (when (and (emitted-equation/emitted-equation? operation)
             ;; New algorithm variants preserve constructor initialization until their own
             ;; complete-write coverage proof is admitted here. A :write use alone is not proof.
             (soac/program-form? (:algorithm operation)))
    (let [algorithm (:algorithm operation)
          facts (soac/facts algorithm)
          equations (soac/equations algorithm)
          destination (get-in facts [:values id])]
      ;; A graph's external write permission does not prove its internal first touch.
      ;; Multi-equation algorithms need an ordered coverage proof before admission.
      (when (and (= 1 (count equations))
                 (= {:kind :plain} (:representation destination))
                 (nil? (:logical-layout destination))
                 (contains? buffers id)
                 (not-any? (fn [input]
                             (and (contains? buffers (:id input))
                                  (= (get buffers id) (get buffers (:id input)))))
                           (filter #(contains? #{:input :inout} (:role %))
                                   (concat (get-in operation [:graph :inputs])
                                           (get-in operation [:graph :outputs])))))
        (some (fn [equation]
                (some (fn [[result physical contract]]
                        (when (and (= id physical) (= :write (:access contract)))
                          (or (coverage/dense-result-covers?
                                algorithm result capacity
                                #(try (first (concrete-shape result {:shape [%]} scalars buffers storage))
                                      (catch clojure.lang.ExceptionInfo error
                                        (if (contains? #{:invocation-link-shape-scalar
                                                         :invocation-link-shape-extent
                                                         :invocation-link-shape-expression
                                                         :unbound-index-symbol
                                                         :index-expression
                                                         :launch-index-expression}
                                                       (:reason (ex-data error)))
                                          nil
                                          (throw error)))))
                              (coverage/rectangular-effect-covers? algorithm result capacity scalars))))
                      (map vector (nth equation 2) (soac/physical-results facts equation)
                           (soac/result-storage facts (second equation)))))
              equations)))))

(defn- complete-write?
  ([operation id capacity scalars buffers storage]
   (complete-write? operation id capacity scalars buffers storage nil))
  ([operation id capacity scalars buffers storage projection]
   (if (equation-dispatch/emitted-equation-dispatch? operation)
     (when-let [extent (get (if projection (:complete-write-domains projection)
                               (equation-dispatch/complete-write-domains operation)) id)]
       (= capacity (launch/resolve-expression #(scalar-number scalars % id) extent)))
     (let [equation (if projection (:boundary projection)
                       (equation-dispatch/boundary-equation operation))]
       (when (emitted-equation/emitted-equation? equation)
         ;; Preserve the existing admission boundary: ordinary SOAC first-touch coverage stays
         ;; alias-aware below. A certified plain contraction domain alone does not promote it.
         (or (when-let [extent (get (if projection
                                       (when (swr/plan? (:algorithm equation))
                                         (:complete-write-domains projection))
                                       (emitted-equation/complete-write-domains equation)) id)]
               (= capacity (launch/resolve-expression #(scalar-number scalars % id) extent)))
             (soac-complete-write? equation id capacity scalars buffers storage)))))))

(defn- equation-boundaries
  "Inspect each immutable operation boundary once within this lowering invocation.
   This retains no validation authority across calls and never selects a runtime alternative."
  ([parallel-program] (equation-boundaries parallel-program nil))
  ([parallel-program projections]
  (into []
        (keep (fn [equation]
                (when-let [operation (first (:operations equation))]
                  (let [projection (when (and projections
                                              (not (emitted-loop/emitted-loop? operation)))
                                     (emitted-program/operation-projection projections operation))]
                    {:operation operation :projection projection
                     :graph (if projection (:graph (:boundary projection))
                                (equation-dispatch/boundary-graph operation))}))))
        (:equations parallel-program))))

(defn- write-before-read-inputs
  [boundaries materialized scalars]
  (let [program-buffers (:program-buffers materialized)
        buffers (into {} (map (fn [[id buffer]] [id (:id buffer)])) program-buffers)
        storage (into {} (map (fn [[_ buffer]] [(:id buffer) buffer])) program-buffers)]
  (into #{}
        (keep (fn [id]
                (let [[access operation projection]
                      (some (fn [{:keys [operation graph projection]}]
                              (when-let [access (graph-buffer-access graph id)]
                                [access operation projection]))
                            boundaries)]
                  (when (and (= :write access)
                             (complete-write? operation id
                                              (reduce *' 1 (:shape (get program-buffers id)))
                                              scalars buffers storage projection)) id))))
        (keys program-buffers))))

(defn- shape-dependencies
  [values abstract]
  (let [declared (set (keys values))]
    (into #{}
          (mapcat
           (fn [dimension]
             (let [projected
                   (index-expression/project-dimension
                    dimension #(get-in values [% :dtype]) fail!)]
               (if-let [backing (launch/backing-extent-id projected)]
                 (when (contains? declared backing) [backing])
                 (set/intersection declared
                                   (launch/expression-references projected))))))
          (:shape abstract))))

(defn- required-materialized-buffers
  "Return materialized runtime values still referenced by the emitted program.

   A graph refinement may eliminate a source-level allocation entirely (for example, product
   partials replaced by workgroup storage). The invocation plan intentionally predates scheduling,
   so it can still materialize that binding. Graph-boundary arrays, host array operands, outputs,
   and their transitive retained shape dependencies survive into the LinkPlan."
  [parallel-program boundaries]
  (let [values (:values parallel-program)
        array-value? #(seq (:shape (get values %)))
        graph-values
        (into #{}
              (mapcat (fn [{:keys [graph]}]
                        (concat (map :id (:inputs graph))
                                (map :id (:outputs graph)))))
              boundaries)
        host-values
        (into #{}
              (comp (filter #(true? (get-in % [:attributes :host-only])))
                    (mapcat :operands)
                    (filter array-value?))
              (:equations parallel-program))
        loop-initials
        (into #{}
              (mapcat (fn [equation]
                        (let [operation (first (:operations equation))]
                          (when (emitted-loop/emitted-loop? operation)
                            (map :initial
                                 (control/carried
                                  (get-in operation [:schedule :algorithm])))))))
              (:equations parallel-program))
        output-values (into #{} (filter array-value?) (:outputs parallel-program))
        initial (into graph-values (concat host-values loop-initials output-values))]
    (loop [required initial]
      (let [dependencies
            (into #{}
                  (mapcat #(shape-dependencies values (get values %)))
                  required)
            expanded (into required dependencies)]
        (if (= required expanded) required (recur expanded))))))

(defn- resident-runtime-interface
  "Partition the required runtime boundary by emitted ABI kind. Rank is consulted only for
   non-ABI shape dependencies, so a logical rank-zero resident buffer remains a buffer."
  [parallel-program boundaries projections]
  (let [boundary-buffer-values
        (into #{}
              (mapcat (fn [{:keys [graph]}]
                        (map :id (concat (:inputs graph) (:outputs graph)))))
              boundaries)
        boundary-scalar-values
        (into #{} (mapcat (comp #(map :id %) :scalars :graph)) boundaries)
        logical-buffer-values
        (reduce
         (fn [buffer-values equation]
           (if-let [operation (first (:operations equation))]
             (let [projection (emitted-program/operation-projection projections operation)
                   physical-results (:physical-results projection)]
               (into buffer-values
                     (keep (fn [result]
                             (when (contains? buffer-values (get physical-results result)) result)))
                     (:results equation)))
             buffer-values))
         boundary-buffer-values
         (:equations parallel-program))
        required-values (required-materialized-buffers parallel-program boundaries)
        _ (when-let [ambiguous (seq (set/intersection logical-buffer-values
                                                       boundary-scalar-values))]
            (fail! :resident-binding-runtime-kind
                   "emitted ABI assigns both buffer and scalar kinds to one program value"
                   {:values (set ambiguous)}))
        non-abi-required (set/difference required-values logical-buffer-values)
        _ (doseq [value-id non-abi-required]
            (when-not (get-in parallel-program [:values value-id])
              (fail! :resident-binding-runtime-value
                     "required resident shape dependency has no retained AbstractValue"
                     {:value value-id})))
        shape-only-buffers
        (into #{}
              (filter #(seq (:shape (get-in parallel-program [:values %]))))
              non-abi-required)
        expected-buffers (into (set/intersection required-values logical-buffer-values)
                               shape-only-buffers)
        _ (when-let [ambiguous (seq (set/intersection expected-buffers
                                                       boundary-scalar-values))]
            (fail! :resident-binding-runtime-kind
                   "required resident storage is also declared as an emitted ABI scalar"
                   {:values (set ambiguous)}))
        expected-scalars (into boundary-scalar-values
                               (set/difference non-abi-required shape-only-buffers))]
    (doseq [value-id expected-scalars]
      (let [value (get-in parallel-program [:values value-id])]
        (when-not (and value (empty? (:shape value)))
          (fail! :resident-binding-runtime-kind
                 "resident scalar binding requires an exact retained rank-zero value"
                 {:value value-id :abstract value}))))
    {:buffers expected-buffers :scalars expected-scalars}))

(defn- add-materialized-inputs
  [state materialized program-values overwrite-inputs required-buffers]
  (reduce-kv
   (fn [state compiler-value buffer]
     (if-not (contains? required-buffers compiler-value)
       state
       (let [token (:id buffer)
             expected (get program-values compiler-value)]
       (when-not expected
         (fail! :invocation-link-program-input
                "materialized buffer names a value absent from the emitted program"
                {:compiler-value compiler-value}))
       (when (and (= :zero (:initialization buffer))
                  (not (contains? overwrite-inputs compiler-value)))
         (fail! :invocation-link-zero-initializer
                "zero-initialized program storage requires an explicit initializer schedule"
                {:compiler-value compiler-value :storage token :shape (:shape buffer)}))
       (let [initialization (if (contains? overwrite-inputs compiler-value)
                              :unspecified
                              (:initialization buffer))]
         (-> state
             (assoc-in [:buffers compiler-value] token)
             (add-storage token compiler-value expected (:shape buffer)
                          (:source buffer) initialization))))))
   state (:program-buffers materialized)))

(defn- lower-loop-storage
  [state invocation-id equation scalars]
  (let [emitted (emitted-loop/validate! (first (:operations equation)))
        algorithm (-> emitted :schedule :algorithm)
        body-storage (soac/physical-result-map (control/body algorithm))]
    (reduce
     (fn [state {:keys [initial parameter result output]}]
       (let [initial-token (get-in state [:buffers initial])
             _ (when-not initial-token
                 (fail! :invocation-link-loop-initial
                        "structured loop carry has no materialized initial storage"
                        {:equation (:id equation) :initial initial :output output}))
             in-place? (= parameter (get body-storage result))
             ;; StructuredControl is a record-backed list IR; use its public outer-value table.
             output-abstract (get (control/outer-values algorithm) output)
             [state output-token]
             (if in-place?
               [(assoc-in state [:buffers output] initial-token) initial-token]
               (allocate-result state invocation-id output output-abstract scalars :loop-output))
             state (if in-place?
                     (update-in state [:storage output-token :compiler-values] conj output)
                     state)]
         (if in-place?
           state
           (let [scratch-token (storage-id invocation-id :loop-alternate output)
                 shape (concrete-shape output output-abstract scalars
                                       (:buffers state) (:storage state))]
             (-> state
                 (assoc-in [:loop-scratch output] scratch-token)
                 (add-storage scratch-token output output-abstract shape nil :unspecified))))))
     state (control/carried algorithm))))

(defn- lower-equation-storage
  [state invocation-id equation program-values scalars physical-projections]
  (cond
    (true? (get-in equation [:attributes :host-only])) state

    (emitted-loop/emitted-loop? (first (:operations equation)))
    (lower-loop-storage state invocation-id equation scalars)

    :else
    (let [operation (first (:operations equation))
          projection (when physical-projections
                       (emitted-program/operation-projection physical-projections operation))
          emitted (if projection (:boundary projection)
                      (equation-dispatch/boundary-equation operation))
          physical (if projection (:physical-results projection)
                       (emitted-equation/physical-results emitted))]
      (reduce
       (fn [state result]
         (let [physical-id (get physical result)
               abstract (or (get program-values physical-id) (get program-values result))
               [state token]
               (allocate-result state invocation-id physical-id abstract scalars :result)]
           (let [logical (get program-values result)
                 shape (concrete-shape result logical scalars (:buffers state) (:storage state))
                 backing (get-in state [:storage token])
                 prefix? (< (reduce *' 1 shape) (reduce *' 1 (:shape backing)))
                 producer (when (soac/program-form? (:algorithm emitted))
                            (some #(when (some #{result} (nth % 2)) %)
                                  (soac/equations (:algorithm emitted))))]
             (if (and prefix? producer (contains? '#{contract map} (soac/operation-kind producer)))
               (let [view-token (storage-id invocation-id :result-view result)]
                 (-> state
                     (add-storage view-token result logical shape nil :unspecified)
                     (assoc-in [:storage view-token :backing] token)
                     (assoc-in [:storage view-token :result-view] [result physical-id])
                     (assoc-in [:buffers result] view-token)))
               (-> state
                   (assoc-in [:buffers result] token)
                   (update-in [:storage token :compiler-values] conj result))))))
       state (:results equation)))))

(defn- storage-link-layout
  "Project selected derived storage through the ordinary node/value constructors. No allocation
   or initialization is performed; callers retain the final plan's memory obligations."
  [{:keys [storage buffers]} scalars target output-tokens tokens]
  {:nodes
   (mapv (fn [token]
           (let [{:keys [shape source abstract backing]} (get storage token)]
             (link/node {:id token :dtype (:dtype abstract) :shape shape :device target
                         :allocation-id (or backing token)
                         :byte-size (when backing
                                      (let [base-shape (get-in storage [backing :shape])]
                                        (bview/required-byte-span (:dtype abstract) base-shape
                                                                  (bview/dense-strides base-shape))))
                         :role (cond
                                 (and source (contains? output-tokens token)) :state
                                 source :state
                                 (contains? output-tokens token) :output
                                 :else :internal)
                         :source source}))) tokens)
   :values
   (mapv (fn [token]
           (let [abstract (get-in storage [token :abstract])]
             (link/value {:id token :abstract (assoc abstract :shape
                                                    (concrete-shape token abstract scalars buffers storage))
                          :leaves [{:name :value :node token}]}))) tokens)})

(defn ^:no-doc ordered-entry-roles
  "Derive conservative semantic entry roles from checked graph accesses plus caller intent.
   This establishes no storage placement, initialization or completion. Written public buffers
   default to resident state; complete-overwrite specialization may refine them separately."
  [state requested-roles]
  (let [requested-roles (or requested-roles {})
        available (materialization/prefix-bindings (:materialization state))
        public (set (keys (:program-buffers available)))
        allowed (disj link/binder-roles :scratch)
        _ (when-not (and (map? requested-roles) (every? allowed (vals requested-roles))
                         (set/subset? (set (keys requested-roles)) public))
            (fail! :invocation-ordered-entry-roles "entry intent must name public buffers with public roles"
                   {:available public :roles requested-roles}))
        _ (when (some #(emitted-loop/emitted-loop? (first (:operations %)))
                       (get-in state [:program :equations]))
            (fail! :invocation-ordered-entry-control
                   "structured control needs its own complete entry-access projection" {}))
        boundaries (equation-boundaries (:program state) (:projections state))
        written (into #{}
                      (filter (fn [id]
                                (some #(contains? #{:write :read-write}
                                                  (graph-buffer-access (:graph %) id)) boundaries)))
                      public)
        roles (merge (into {} (map (fn [id] [id (if (contains? written id) :state :input)])) public)
                     requested-roles)]
    (doseq [id written :let [role (get roles id)] :when (link/read-only-node-role? role)]
      (fail! :link-write-read-only "a program cannot write a read-only public entry"
             {:value id :role role}))
    roles))

(defn- validate-ordered-entry-aliases! [buffers roles]
  (when-let [[left right]
             (materialization/conflicting-source-alias
              (map (fn [[id buffer]]
                     {:source (:source buffer) :storage (:id buffer) :value id
                      :mutable? (not (link/read-only-node-role? (get roles id)))}) buffers))]
    (fail! :invocation-ordered-source-alias
           "mutable host aliases require one explicit resident storage identity, not independent copies"
           {:values #{(:value left) (:value right)}})))

(defn ^:no-doc ordered-entry-layout
  "Describe captured entry buffers without evaluating a source action or granting completion.
   Dynamic inputs remain unacquired; source-created arrays are not captured entry storage."
  [state roles]
  (let [available (materialization/prefix-bindings (:materialization state))
        _ (validate-ordered-entry-aliases! (:program-buffers available) roles)
        captured (into #{} (keep (fn [[id role]] (when (not= :input role) id))) roles)
        realized (add-materialized-inputs {:buffers {} :storage {} :loop-scratch {}}
                                          available (get-in state [:program :values]) #{} captured)
        scalars (merge (invocation-shape-scalars (:materialization state)) (:program-scalars available))
        refresh (reduce-kv
                 (fn [result id buffer]
                   (let [token (:id buffer) role (get roles id)]
                     (when (or (nil? role)
                               (and (contains? result token) (not= role (get result token))))
                       (fail! :invocation-public-role-alias
                              "entry storage requires one consistent role for all its public aliases"
                              {:value id :storage token}))
                     (assoc result token role))) {} (:program-buffers available))
        layout (storage-link-layout realized scalars (:target state) #{} (keys (:storage realized)))]
    (assoc layout :nodes (mapv #(assoc % :role (get refresh (:id %))) (:nodes layout))
                  :entry-refresh refresh)))

(defn- terminal-export-contracts
  [semantic-program program projections]
  (when-let [contract (get-in semantic-program [:attributes :invocation-plan :attributes :terminal-host-result])]
    (mapv
     (fn [{:keys [value dtype producer-equation result-position] :as parameter}]
       (let [equation (some #(when (= producer-equation (:id %)) %) (:equations program))
             operation (first (:operations equation))
             projection (when operation (emitted-program/operation-projection projections operation))
             physical (get-in projection [:physical-results value])
             graph (get-in projection [:boundary :graph])
             output (some #(when (= physical (:id %)) %) (:outputs graph))]
         (when-not (and equation (not (true? (get-in equation [:attributes :host-only])))
                        (= 1 (count (:operations equation)))
                        (= value (nth (:results equation) result-position nil))
                        (= dtype (:dtype output)) (= 1 (:elements output)))
           (fail! :invocation-terminal-export
                  "terminal parameter lacks an exact one-element emitted output ABI"
                  {:parameter parameter :physical physical :output output}))
         (assoc parameter :physical-value physical)))
     (:parameters contract))))

(defn ^:no-doc begin-ordered-realization
  "Begin owner-local realization without evaluating invocation steps or acquiring resources.
   The returned map is runtime construction state, not a portable completion certificate."
  [compilation arguments caller-options]
  (let [semantic-program (:semantic compilation)
        invocation (get-in semantic-program [:attributes :invocation-plan])
        invocation (invocation/validate-against! invocation semantic-program)
        order (invocation/source-order invocation semantic-program)
        checked (if (nil? caller-options)
                  (emitted-program/validate-with-physical-results! (:emitted compilation))
                  (emitted-program/validate-with-physical-results! (:emitted compilation) caller-options))
        program (:program checked)
        boundary #(mapv (fn [equation]
                          (select-keys equation [:id :operands :results :algorithm]))
                        (:equations %))]
    (when-not (and (semantic/equivalent? (boundary semantic-program) (boundary program))
                   (semantic/equivalent? (:values semantic-program)
                                         (select-keys (:values program)
                                                      (keys (:values semantic-program)))))
      (fail! :invocation-ordered-program
             "ordered realization requires the exact retained semantic and emitted boundary" {}))
    {:program program :source-order order :source-cursor 0
     :terminal-exports (terminal-export-contracts semantic-program program (:projections checked))
     :source-realizations (into {} (keep (fn [entry]
                                          (when-let [action (invocation-source/realization-action semantic-program entry)]
                                            [(:position entry) action]))) order)
     :materialization (materialization/begin-materialization invocation arguments)
     :realized {:buffers {} :storage {} :loop-scratch {}}
     :acquired-storage #{}
     :source-allocations (into {} (map (juxt :destination :source-expression))
                              (get-in semantic-program [:attributes :allocations]))
     :projections (:projections checked) :target (:target compilation)
     :caller-options caller-options}))

(defn- validate-source-result-allocation! [state entry action]
  (let [destination (:binding action)
        equation (some #(when (= (:equation action) (:id %)) %) (get-in state [:program :equations]))
        operation (first (:operations equation))
        projection (when operation
                     (emitted-program/operation-projection (:projections state) operation))]
    ;; Prefix-result views and distinct logical/physical storage need their own reconciliation.
    (when-not (and (symbol? destination)
                   (contains? (:physical-results projection) destination)
                   (= destination (get-in projection [:physical-results destination])))
      (fail! :invocation-ordered-realization-storage
             "source allocation must be the equation's exact physical result"
             {:entry entry :destination destination}))))

(defn ^:no-doc validate-ordered-source-actions!
  "Check structural executor coverage after begin-ordered-realization, before capture or source
   effects. This evaluates no source operand, acquires no storage and grants no completion proof.
   Runtime counts, initialization, input ownership and logical-view checks remain separate."
  [state]
  (when-let [providers (seq (get-in state [:program :attributes :native-initialization-providers]))]
    (fail! :invocation-link-native-initialization
           "ordered linking requires an explicit native-content provider"
           {:destinations (vec providers)}))
  (let [steps (into {} (map (juxt :id identity)) (get-in state [:materialization :plan :steps]))
        equations (into {} (map (juxt :id identity)) (get-in state [:program :equations]))]
    (doseq [entry (:source-order state)
            :let [step (get steps (:invocation-step entry))
                  equation (get equations (:equation entry))
                  action (get (:source-realizations state) (:position entry))]]
      (when (and (:terminal-result entry)
                 (not (and (= [:return] (:terminal-result entry))
                           (= (:position entry) (dec (count (:source-order state))))
                           (seq (:terminal-exports state)))))
        (fail! :invocation-ordered-source-action "invalid terminal observation entry" {:entry entry}))
      (when (or (and (:realization entry)
                     (not (contains? #{:result-allocation :storage-alias} (:kind action))))
                (and step (not (or (invocation/scalar-compute? step)
                                         (invocation/shape-projection? step)
                                         (and (:storage entry) (invocation/buffer-allocation? step))))))
        (fail! :invocation-ordered-source-action
               "ordered source entry requires an unimplemented realization action" {:entry entry}))
      (when (some emitted-loop/emitted-loop? (:operations equation))
        (fail! :invocation-ordered-source-action
               "ordered execution does not yet admit structured-loop fragments" {:entry entry}))
      (when (= :result-allocation (:kind action))
        (validate-source-result-allocation! state entry action))))
  state)

(defn ^:no-doc terminal-export-layout
  "Resolve terminal exports from already realized producer storage. This acquires no storage
   and supplies no completion evidence; the execution owner must prove initialized views."
  [state]
  (let [entry (nth (:source-order state) (:source-cursor state) nil)]
    (when-not (and (:terminal-result entry) (nil? (:pending state))
                   (= (:source-cursor state) (dec (count (:source-order state)))))
      (fail! :invocation-terminal-order "terminal observation is not the next completed source boundary" {}))
    (let [exports
          (mapv
           (fn [{:keys [value physical-value dtype] :as export}]
             (let [token (get-in state [:realized :buffers value])
                   physical-token (get-in state [:realized :buffers physical-value])
                   storage (get-in state [:realized :storage token])]
               (when-not (and token (= token physical-token)
                              (contains? (:acquired-storage state) token)
                              (= dtype (get-in storage [:abstract :dtype]))
                              (= 1 (reduce *' 1 (:shape storage)))
                              (nil? (:source storage)) (nil? (:backing storage)))
                 (fail! :invocation-terminal-storage
                        "terminal export differs from acquired producer storage" {:export export}))
               (assoc export :node token)))
           (:terminal-exports state))
          layout (storage-link-layout (:realized state) {} (:target state) #{}
                                      (vec (distinct (map :node exports))))]
      (assoc layout :exports exports))))

(defn ^:no-doc ordered-input-arguments
  "Preflight primitive host-array replacements without evaluating source steps or mutating an
   owner. Only dynamic entry inputs may change, and the prepared storage specialization is fixed.
   Aggregate and resident-device replacement require their own ownership-aware adapter."
  [state arguments roles replacements]
  (when-not (map? replacements)
    (fail! :invocation-ordered-inputs "ordered input replacement requires a map" {}))
  (let [plan (get-in state [:materialization :plan])
        _ (when (get-in plan [:attributes :parameter-projection])
            (fail! :invocation-ordered-input-projection
                   "aggregate input replacement requires an ownership-aware field adapter" {}))
        available (materialization/prefix-bindings (:materialization state))
        buffers (:program-buffers available)
        input-ids (into #{} (keep (fn [[id role]] (when (= :input role) id))) roles)
        _ (when-not (set/subset? (set (keys replacements)) input-ids)
            (fail! :invocation-ordered-input-role
                   "only dynamic public inputs may be replaced; state and constants are captured"
                   {:inputs input-ids :supplied (set (keys replacements))}))
        slots (into {} (map-indexed (fn [index parameter] [(:id parameter) index])) (:parameters plan))
        arguments (reduce-kv
                   (fn [arguments id source]
                     (let [slot (get slots (get-in buffers [id :id]))]
                       (when-not (some? slot)
                         (fail! :invocation-ordered-input-slot
                                "dynamic input requires an exact public parameter slot" {:value id}))
                       (assoc arguments slot source)))
                   (vec arguments) replacements)
        checked (materialization/prefix-bindings (materialization/begin-materialization plan arguments))]
    (validate-ordered-entry-aliases! (:program-buffers checked) roles)
    (doseq [[id buffer] buffers
            :let [actual (get-in checked [:program-buffers id])]]
      (when-not (and (= (:id buffer) (:id actual)) (= (:shape buffer) (:shape actual))
                     (av/storage-contract-compatible? (:value buffer) (:value actual)))
        (fail! :invocation-ordered-input-specialization
               "input replacement cannot change the prepared storage specialization" {:value id})))
    arguments))

(defn ^:no-doc advance-host-source
  "Evaluate the next host source definition once, including shared invocation/equation producers.
   This owner-local transition never acquires storage or acknowledges numerical completion."
  [state evaluate-step evaluate-host caller-options]
  (when-not (= caller-options (:caller-options state))
    (fail! :invocation-ordered-request "ordered realization cannot change its caller math request" {}))
  (when (:pending state)
    (fail! :invocation-ordered-pending "the preceding source action has not completed" {}))
  (let [entry (nth (:source-order state) (:source-cursor state) nil)
        materialized (materialization/validate-state! (:materialization state))
        step (when (:invocation-step entry)
               (nth (get-in materialized [:plan :steps]) (:next-step materialized) nil))
        equation (when (:equation entry)
                   (some #(when (= (:equation entry) (:id %)) %) (get-in state [:program :equations])))]
    (when-not (and entry (not (:storage entry)) (not (:realization entry))
                   (or (not (:invocation-step entry))
                       (and (= (:invocation-step entry) (:id step))
                            (or (invocation/scalar-compute? step)
                                (invocation/shape-projection? step))))
                   (or (not (:equation entry))
                       (and equation (true? (get-in equation [:attributes :host-only]))))
                   (or step equation))
      (fail! :invocation-ordered-next-host
             "the next source entry is not an available scalar host definition" {:entry entry}))
    (let [advanced (volatile! materialized)
          run-step! (fn []
                      (vreset! advanced
                               (link/without-validation-context
                                #(materialization/materialize-next materialized evaluate-step))))
          available (materialization/prefix-bindings materialized)
          scalars (merge (invocation-shape-scalars materialized) (:program-scalars available)
                         (:host-scalars state))
          buffers (reduce-kv
                   (fn [bindings id buffer]
                     (let [token (:id buffer)]
                       (when (and (contains? bindings id) (not= token (get bindings id)))
                         (fail! :invocation-ordered-host-buffer
                                "available invocation buffer differs from its realized binding"
                                {:value id :available token :realized (get bindings id)}))
                       (assoc bindings id token)))
                   (get-in state [:realized :buffers]) (:program-buffers available))
          buffer-shapes (into {} (map (fn [[id buffer]] [id (:shape buffer)]))
                              (:program-buffers available))
          ;; A shared producer is evaluated by the invocation interpreter inside the checked
          ;; host callback. The host constructor validates that same result, not a second eval.
          host-step (when equation
                      (program-call/make-host-equation-step
                       (:program state) (:id equation) buffers scalars
                       (if step
                         (fn [_ _] {(first (:results equation))
                                    (get-in (run-step!) [:values (:id step)])})
                         (when evaluate-host
                           (fn [equation context]
                             (evaluate-host equation (assoc context :buffer-shapes buffer-shapes)))))
                       caller-options))
          _ (when (and step (not equation)) (run-step!))]
      (cond-> (-> state
                  (assoc :materialization @advanced)
                  (update :source-cursor inc))
        host-step (update :host-scalars merge (:results host-step))
        host-step (assoc-in [:host-steps (:id equation)] host-step)))))

(defn ^:no-doc prepare-source-allocation
  "Prepare the exact next source allocation, not an unspecified numerical result buffer.
   evaluate-size is the trusted typed host evaluator returning the source value and its actual
   constructor count. Acquisition and zero initialization must complete before cursor advance."
  [state evaluate-size caller-options]
  (when-not (= caller-options (:caller-options state))
    (fail! :invocation-ordered-request "ordered realization cannot change its caller math request" {}))
  (when (:pending state)
    (fail! :invocation-ordered-pending "the preceding source action has not completed" {}))
  (let [entry (nth (:source-order state) (:source-cursor state) nil)
        action (get (:source-realizations state) (:source-cursor state))
        auxiliary? (= :result-allocation (:kind action))
        destination (or (:storage entry) (when auxiliary? (:binding action)))
        expression (if auxiliary? (:expression action) (get (:source-allocations state) destination))
        abstract (get-in state [:program :values destination])
        values (get-in state [:program :values])
        projection (invocation-source/allocation-source expression values)
        step (some #(when (= (:invocation-step entry) (:id %)) %)
                   (get-in state [:materialization :plan :steps]))
        initialization (:initialization projection)]
    (when-not (and destination (= :zero initialization)
                   (invocation-source/allocation-corresponds? expression abstract initialization values))
      (fail! :invocation-ordered-next-allocation
             "the next source entry is not an admitted size-only zero allocation" {:entry entry}))
    (when auxiliary? (validate-source-result-allocation! state entry action))
    (when (and (:helper projection)
               (not (semantic/equivalent? (select-keys projection [:constructor :helper])
                                         (:source-projection step))))
      (fail! :invocation-ordered-allocation-definition
             "allocation helper differs from this invocation's retained constructor definition"
             {:entry entry}))
    (when-not (ifn? evaluate-size)
      (fail! :invocation-ordered-size-evaluator
             "source allocation requires the typed source-size evaluator" {:entry entry}))
    (let [materialization (materialization/validate-state! (:materialization state))
          available (materialization/prefix-bindings materialization)
          scalars (merge (invocation-shape-scalars materialization)
                         (:program-scalars available) (:host-scalars state))
          _ (doseq [argument (get-in projection [:helper :arguments])]
              (when-not (or (contains? scalars argument)
                            (contains? (:program-buffers available) argument)
                            (contains? (:acquired-storage state)
                                       (get-in state [:realized :buffers argument])))
                (fail! :invocation-ordered-allocation-argument
                       "allocation helper arguments must already have checked source values or acquired storage"
                       {:argument argument :entry entry})))
          extent (:extent projection)
          {:keys [source-value elements]} (link/without-validation-context #(evaluate-size extent scalars))]
      (when-not (and (integer? elements) (<= 0 elements Integer/MAX_VALUE))
        (fail! :invocation-ordered-allocation-count
               "source allocation evaluator must return a nonnegative constructor count" {}))
      ;; Retained dimensions currently describe integral mathematical extents. A wrapping
      ;; Number.intValue conversion cannot silently become that many native elements.
      (when-not (and (integer? source-value) (= source-value elements))
        (fail! :invocation-ordered-allocation-shape
               "source constructor count cannot be reconciled with the retained extent"
               {:source-value source-value :constructor-count elements :destination destination}))
      (let [shape (concrete-shape destination abstract scalars
                                  (get-in state [:realized :buffers]) (get-in state [:realized :storage]))]
        (when-not (= [elements] shape)
          (fail! :invocation-ordered-allocation-shape
                 "retained storage differs from the evaluated source constructor count"
                 {:shape shape :constructor-count elements :destination destination}))
        (let [step (when-let [id (:invocation-step entry)]
                     (let [next-step (nth (get-in materialization [:plan :steps])
                                          (:next-step materialization) nil)]
                       (when-not (and (= id (:id next-step))
                                      (invocation/buffer-allocation? next-step))
                         (fail! :invocation-ordered-allocation-step
                                "source allocation differs from the next invocation step" {:entry entry}))
                       next-step))
              materialization (if step (materialization/materialize-next materialization nil)
                                  materialization)
              [realized token]
              (if step
                (let [buffer (get-in materialization [:values (:id step)])
                      token (:id buffer)]
                  [(-> (:realized state)
                       (assoc-in [:buffers destination] token)
                       (add-storage token destination abstract shape nil :zero)) token])
                (allocate-result (:realized state) (get-in materialization [:plan :id])
                                 destination abstract scalars :result))
              _ (when (contains? (:acquired-storage state) token)
                  (fail! :invocation-ordered-allocation-acquired
                         "source allocation is already acquired in this invocation" {:storage token}))
              realized (assoc-in realized [:storage token :initialization] :zero)
              layout (storage-link-layout realized scalars (:target state) #{token} [token])]
          {:state (assoc state :realized realized :materialization materialization
                               :pending {:kind :source-allocation :storage token
                                         :source-cursor (:source-cursor state)})
           :nodes (:nodes layout) :values (:values layout)
           :storage token :initialization :zero :elements elements})))))

(defn ^:no-doc advance-source-alias
  "Bind one validated source alias to already acquired storage. This neither acquires storage
   nor grants initialization or producer-completion credit."
  [state caller-options]
  (when-not (= caller-options (:caller-options state))
    (fail! :invocation-ordered-request "ordered realization cannot change its caller math request" {}))
  (when (:pending state)
    (fail! :invocation-ordered-pending "the preceding source action has not completed" {}))
  (let [entry (nth (:source-order state) (:source-cursor state) nil)
        {:keys [kind binding expression]} (get (:source-realizations state) (:source-cursor state))
        realized (:realized state)
        token (get-in realized [:buffers expression])
        existing (get-in realized [:buffers binding])]
    (when-not (and (= :storage-alias kind) (= binding (:binding entry))
                   (symbol? expression) token (contains? (:acquired-storage state) token)
                   (or (nil? existing) (= existing token)))
      (fail! :invocation-ordered-alias-storage
             "source alias requires the exact already acquired destination storage"
             {:entry entry :source expression :storage token}))
    (let [available (materialization/prefix-bindings (:materialization state))
          scalars (merge (invocation-shape-scalars (:materialization state))
                         (:program-scalars available) (:host-scalars state))
          abstract (get-in state [:program :values binding])
          shape (concrete-shape binding abstract scalars (:buffers realized) (:storage realized))]
      (-> state
          (assoc :realized (-> realized
                              (assoc-in [:buffers binding] token)
                              (add-storage token binding abstract shape nil :unspecified)))
          (update :source-cursor inc)))))

(defn ^:no-doc realize-ready-equation
  "Realize only the next numerical source entry into an ordinary concrete graph LinkPlan.
   This neither executes nor advances the cursor. Its conditional entry contract must be
   satisfied by the enclosing owner, and produced views become ready only after completion."
  [state equation-id caller-options]
  (when-not (= caller-options (:caller-options state))
    (fail! :invocation-ordered-request "ordered realization cannot change its caller math request" {}))
  (when (:pending state)
    (fail! :invocation-ordered-pending "the preceding numerical stage has not completed" {}))
  (let [entry (nth (:source-order state) (:source-cursor state) nil)
        program (:program state)
        equation (some #(when (= equation-id (:id %)) %) (:equations program))]
    (when-not (and equation (= equation-id (:equation entry))
                   (not (true? (get-in equation [:attributes :host-only])))
                   (not (emitted-loop/emitted-loop? (first (:operations equation)))))
      (fail! :invocation-ordered-next-equation
             "only the next numerical source entry may acquire ready storage"
             {:entry entry :equation equation-id}))
    (let [materialization (materialization/validate-state! (:materialization state))
          available (materialization/prefix-bindings materialization)
          scalars (merge (invocation-shape-scalars materialization)
                         (:program-scalars available) (:host-scalars state))
          boundaries (equation-boundaries {:equations [equation]} (:projections state))
          required (required-materialized-buffers
                    {:values (:values program) :equations [equation] :outputs (:results equation)}
                    boundaries)
          overwrite (write-before-read-inputs boundaries available scalars)
          acquired (:acquired-storage state)
          unacquired (update available :program-buffers
                             (fn [buffers]
                               (into {} (remove (fn [[id buffer]]
                                                  (when (contains? acquired (:id buffer))
                                                    (when-not (= (:id buffer)
                                                                 (get-in state [:realized :buffers id]))
                                                      (fail! :invocation-ordered-input-storage
                                                             "acquired input lost its logical storage binding"
                                                             {:value id}))
                                                    true))) buffers)))
          realized (add-materialized-inputs (:realized state) unacquired (:values program)
                                            overwrite required)
          realized (lower-equation-storage realized (get-in materialization [:plan :id]) equation
                                            (:values program) scalars (:projections state))
          result-views (into {} (keep :result-view) (vals (:storage realized)))
          {:keys [call buffers]} (program-call/make-equation-call
                                 program equation-id (:buffers realized) scalars result-views caller-options)
          _ (when (seq (:result-views call))
              (fail! :invocation-ordered-result-views
                     "ready graph stages must retain logical prefix-result contracts before admission"
                     {:equation equation-id :result-views (:result-views call)}))
          realized (assoc realized :buffers buffers)
          outputs (vec (distinct (vals (:outputs call))))
          tokens (vec (distinct (concat (vals (:buffers call)) outputs)))
          layout (storage-link-layout realized scalars (:target state) (set outputs) tokens)
          nodes (mapv (fn [node]
                        (if (contains? acquired (get-in node [:view :allocation :id]))
                          ;; Existing roots are conditional inputs to this stage. Retaining
                          ;; their old host initializer would silently overwrite earlier work.
                          (assoc node :source nil :role (if (some #{(:id node)} outputs) :state :input))
                          node))
                      (:nodes layout))
          instance-id [:ordered-equation equation-id]
          instance (link/graph-instance
                    {:id instance-id :graph (:graph call) :bindings (:buffers call)
                     :scalar-values (:scalar-values call)
                     :attributes {:emitted-equation-source {:program program :equation-id equation-id}}})
          request {:id [(get-in materialization [:plan :id]) :stage equation-id]
                   :target (:target state) :nodes nodes :values (:values layout)
                   :instances [instance] :outputs outputs
                   :aliases (into #{} (map set)
                                  (bview/overlapping-id-pairs (map (juxt :id :view) nodes)))}
          {:keys [plan effect-evidence]}
          (if (nil? caller-options)
            (link/make-with-effect-evidence request)
            (link/make-with-effect-evidence request caller-options))
          contract (first (link/relative-entry-contracts! plan effect-evidence
                                                         [[instance-id]] caller-options))]
      {:state (assoc state :realized realized
                           :pending {:equation equation-id :source-cursor (:source-cursor state)})
       :call call :stage-plan plan :effect-evidence effect-evidence :entry-contract contract
       :new-storage (select-keys (:storage realized)
                                 (remove #(contains? acquired %) tokens))})))

(defn- lower-in-context
  "Lower one specialized invocation and matching emitted program into a validated LinkPlan.

   `evaluate-host` is passed unchanged to EmittedParallelProgramCall for closed, effect-free host
   scalar equations. The returned plan is allocation-free; runtime contact starts only in
   raster.gpu.link/instantiate!.
   Retained validation is internal exact-owner static evidence, not a runtime proof;
   materialization, call bindings and final LinkPlan memory/effect obligations remain fresh.
   Only unchanged static program validation can be shared with final construction."
  [materialized parallel-program target evaluate-host project retained-validation caller-options
   & [source-state]]
  (let [materialized (materialization/validate! materialized)
        _ (when (and (get-in materialized [:plan :attributes :terminal-host-result])
                     (not (:terminal-observed? source-state)))
            (fail! :explicit-host-result-required
                   "terminal observation cannot be lowered as an eager resident invocation" {}))
        ;; Without a retained owner, independently prove this exact program once for this
        ;; synchronous lowering. The same sealed report feeds the storage/call/final phases;
        ;; it is never published as authority on the returned plan or bound around callbacks.
        retained-validation (if retained-validation
                              (if (nil? caller-options)
                                (emitted-program/checked-retained-validation! parallel-program retained-validation)
                                (emitted-program/checked-retained-validation! parallel-program retained-validation caller-options))
                              (if (nil? caller-options)
                                (emitted-program/validate-with-physical-results! parallel-program)
                                (emitted-program/validate-with-physical-results! parallel-program caller-options)))
        parallel-program (:program retained-validation)
        _ (when-let [providers (seq (get-in parallel-program
                                           [:attributes :native-initialization-providers]))]
            (fail! :invocation-link-native-initialization
                   "source-independent linking requires an explicit native-content provider"
                   {:destinations (vec providers)}))
        invocation-plan (:plan materialized)
        invocation-id (:id invocation-plan)
        _ (when-not (set/subset? (set (:inputs parallel-program))
                                 (set (:program-inputs invocation-plan)))
            (fail! :invocation-link-program-boundary
                   "emitted program introduced inputs outside its semantic invocation boundary"
                   {:invocation-inputs (:program-inputs invocation-plan)
                    :program-inputs (:inputs parallel-program)}))
        scalars (:program-scalars materialized)
        shape-scalars (merge (invocation-shape-scalars materialized) scalars (:host-scalars source-state))
        boundaries (equation-boundaries parallel-program (:projections retained-validation))
        overwrite-inputs (write-before-read-inputs boundaries
                                                   materialized shape-scalars)
        required-buffers (required-materialized-buffers parallel-program boundaries)
        initial (when-not source-state
                  (add-materialized-inputs {:buffers {} :loop-scratch {} :storage {}}
                                           materialized (:values parallel-program)
                                           overwrite-inputs required-buffers))
        realized (if source-state
                   (:realized source-state)
                   (reduce #(lower-equation-storage %1 invocation-id %2
                                                    (:values parallel-program) shape-scalars
                                                    (:projections retained-validation))
                           initial (:equations parallel-program)))
        call-scalars (select-keys shape-scalars (keys (:values parallel-program)))
        ;; Shape-only invocation scalars are part of the physical program contract even when no
        ;; kernel ABI consumes them. Retain them in the call so logical N-D values can be proved
        ;; element-equivalent to flattened result storage at the LinkPlan boundary.
        call (if (nil? caller-options)
               (program-call/make parallel-program (:buffers realized) call-scalars
                                (:loop-scratch realized) evaluate-host
                                (into {} (keep :result-view) (vals (:storage realized)))
                                retained-validation)
               (program-call/make parallel-program (:buffers realized) call-scalars
                                  (:loop-scratch realized) evaluate-host
                                  (into {} (keep :result-view) (vals (:storage realized)))
                                  retained-validation caller-options))
        resident-outputs (into [] (remove (comp typed-scalar? val)) (:outputs call))
        output-tokens (set (map val resident-outputs))
        storage (:storage realized)
        public-buffer-bindings
        (into {}
              (keep (fn [{:keys [id symbol]}]
                      (let [materialized-value (get-in materialized [:values id])
                            token (when (materialization/materialized-buffer? materialized-value)
                                    (:id materialized-value))]
                        (when (contains? storage token) [symbol token]))))
              (:parameters invocation-plan))
        overwrite-tokens
        (into #{} (keep #(get-in realized [:buffers %])) overwrite-inputs)
        written-program-values
        (into #{}
              (filter
               (fn [compiler-value]
                 (some (fn [{:keys [graph]}]
                         (contains? #{:write :read-write}
                                    (graph-buffer-access graph compiler-value)))
                       boundaries)))
              (keys (:buffers call)))
        written-tokens
        (into #{} (keep #(get-in realized [:buffers %])) written-program-values)
        public-buffer-roles
        (into {}
              (map (fn [[symbol token]]
                     [symbol (cond
                               (contains? overwrite-tokens token) :output
                               (or (contains? written-tokens token)
                                   (contains? output-tokens token)) :state
                               :else :input)]))
              public-buffer-bindings)
        ;; Preserve resolved logical shapes; composition must not rediscover dimensions.
        {:keys [nodes values]}
        (storage-link-layout realized shape-scalars target output-tokens (keys storage))
        ;; Roles and escaped outputs are projected before the final LinkPlan proof.
        ;; Keep this exact checked call inside an unescaped instance candidate;
        ;; validating its pre-projection role map would recheck the same program
        ;; and cannot certify the projected instance anyway.
        instance (link/program-instance-candidate
                  {:id [invocation-id :emitted-program] :call call
                   :attributes {:source :typed-invocation}})]
    (let [request {:id [invocation-id :link-plan]
      :target target
      :nodes nodes
      :values values
      :instances [instance]
      :aliases (into #{} (map set)
                     (bview/overlapping-id-pairs
                      (map (juxt :id :view) nodes)))
      :outputs (mapv val resident-outputs)
      :attributes {:source :typed-invocation
                   :invocation-plan invocation-id
                   :compiler-buffer-bindings (:buffers realized)
                   :public-buffer-bindings public-buffer-bindings
                   :public-buffer-roles public-buffer-roles
                   :semantic-outputs resident-outputs
                   :host-outputs (into {} (filter (comp typed-scalar? val)) (:outputs call))
                   :driver-allocations 0}}]
      (if (nil? caller-options)
        (link/make-with-final-projection request project retained-validation)
        (link/make-with-final-projection request project retained-validation caller-options)))))

(defn- lower-for-request
  [materialized parallel-program target evaluate-host project retained-validation caller-options]
  (link/without-validation-context
   #(lower-in-context materialized parallel-program target evaluate-host project
                      retained-validation caller-options)))

(def ^:private resident-binding-keys
  #{:id :target :nodes :values :buffer-bindings :scalar-values})

(defn- resident-single-leaf-storage
  [nodes values program-values buffer-bindings]
  (reduce-kv
   (fn [state compiler-value value-id]
     (let [expected (get program-values compiler-value)
           link-value (get values value-id)
           leaves (:leaves link-value)]
       (when-not expected
         (fail! :resident-binding-compiler-value
                "resident binding names a value absent from the emitted program"
                {:compiler-value compiler-value}))
       (when-not link-value
         (fail! :resident-binding-link-value
                "resident binding names an absent logical LinkValue"
                {:compiler-value compiler-value :value value-id}))
       (when-not (= 1 (count leaves))
         (fail! :resident-binding-leaf-count
                "one emitted program buffer requires one whole single-leaf LinkValue"
                {:compiler-value compiler-value :value value-id :leaves leaves}))
       (let [node-id (get-in leaves [0 :node])
             node (get nodes node-id)]
         (when-not node
           (fail! :resident-binding-node
                  "resident binding LinkValue names an absent physical node"
                  {:compiler-value compiler-value :value value-id :node node-id}))
         (when-not (av/storage-contract-compatible? expected (:abstract link-value))
           (fail! :resident-binding-value-contract
                  "emitted program value differs from its supplied logical LinkValue"
                  {:compiler-value compiler-value :value value-id
                   :expected expected :actual (:abstract link-value)}))
         (add-storage state value-id compiler-value (:abstract link-value)
                      (get-in node [:view :shape]) nil :unspecified))))
   {:buffers buffer-bindings :loop-scratch {} :storage {}}
   buffer-bindings))

(defn- resident-program-context!
  [parallel-program request caller-options]
  (when-not (and (map? request) (= resident-binding-keys (set (keys request))))
    (fail! :resident-binding-request
           "resident program binding requires one closed storage request"
           {:expected resident-binding-keys
            :actual (when (map? request) (set (keys request)))}))
  (let [{:keys [id target nodes values buffer-bindings scalar-values]} request
        _ (when (nil? id)
            (fail! :resident-binding-id "resident program binding requires a stable identity" {}))
        _ (when-not (and (map? nodes) (map? values)
                         (map? buffer-bindings) (map? scalar-values))
            (fail! :resident-binding-shape
                   "resident program nodes, values, buffers and scalars must be maps"
                   {:nodes (type nodes) :values (type values)
                    :buffer-bindings (type buffer-bindings)
                    :scalar-values (type scalar-values)}))
        retained-validation
        (if (nil? caller-options)
          (emitted-program/validate-with-physical-results! parallel-program)
          (emitted-program/validate-with-physical-results! parallel-program caller-options))
        parallel-program (:program retained-validation)
        _ (when-let [providers (seq (get-in parallel-program
                                             [:attributes :native-initialization-providers]))]
            (fail! :resident-binding-native-initialization
                   "resident program binding cannot supply native initialization content"
                   {:destinations (vec providers)}))
        host-equations (filterv #(true? (get-in % [:attributes :host-only]))
                                (:equations parallel-program))
        _ (when (seq host-equations)
            (fail! :resident-binding-host-equations
                   "resident program binding cannot evaluate host equations"
                   {:equations (mapv :id host-equations)}))
        loop-equations (filterv #(emitted-loop/emitted-loop?
                                  (first (:operations %)))
                                (:equations parallel-program))
        _ (when (seq loop-equations)
            (fail! :resident-binding-structured-loop
                   "the first resident binding slice does not allocate loop carry scratch"
                   {:equations (mapv :id loop-equations)}))
        boundaries (equation-boundaries parallel-program (:projections retained-validation))
        {expected-buffers :buffers expected-scalars :scalars}
        (resident-runtime-interface parallel-program boundaries
                                    (:projections retained-validation))
        _ (when-not (= expected-scalars (set (keys scalar-values)))
            (fail! :resident-binding-scalars
                   "resident scalar bindings must exactly match scalar program inputs"
                   {:expected expected-scalars :actual (set (keys scalar-values))}))]
    {:program parallel-program :retained-validation retained-validation
     :expected-buffers expected-buffers}))

(defn- bind-resident-program-in-context
  [parallel-program request caller-options]
  (let [{:keys [program retained-validation expected-buffers]}
        (resident-program-context! parallel-program request caller-options)
        parallel-program program
        {:keys [id target nodes values buffer-bindings scalar-values]} request
        _ (when-not (= expected-buffers (set (keys buffer-bindings)))
            (fail! :resident-binding-buffers
                   "resident buffer bindings must exactly cover the emitted program boundary"
                   {:expected expected-buffers :actual (set (keys buffer-bindings))}))
        ;; This check is deliberately structural only. Sources and roles are not consulted as
        ;; initialization authority; the final LinkPlan/LinkModule proves ordered readiness.
        initial (resident-single-leaf-storage nodes values (:values parallel-program)
                                              buffer-bindings)
        realized (reduce #(lower-equation-storage %1 id %2 (:values parallel-program)
                                                  scalar-values (:projections retained-validation))
                         initial (:equations parallel-program))
        _ (when-not (and (= buffer-bindings (:buffers realized))
                         (empty? (:loop-scratch realized))
                         (= (:storage initial) (:storage realized))
                         (= (set (vals buffer-bindings))
                            (set (keys (:storage realized))))
                         (every? #(and (nil? (:backing %)) (nil? (:result-view %)))
                                 (vals (:storage realized))))
            (fail! :resident-binding-storage-rewrite
                   "resident traversal attempted to allocate or rewrite supplied storage"
                   {:before buffer-bindings :after (:buffers realized)
                    :loop-scratch (:loop-scratch realized)
                    :storage (set (keys (:storage realized)))}))
        call (if (nil? caller-options)
               (program-call/make parallel-program buffer-bindings scalar-values {} nil {}
                                  retained-validation)
               (program-call/make parallel-program buffer-bindings scalar-values {} nil {}
                                  retained-validation caller-options))
        instance (if (nil? caller-options)
                   (link/program-instance {:id id :call call
                                           :attributes {:source :resident-emitted-program}})
                   (link/program-instance {:id id :call call
                                           :attributes {:source :resident-emitted-program}}
                                          caller-options))]
    ;; Run the complete graph ABI/bounded-loop/storage checker without effect initialization.
    (link/validate-program-instance-storage! target nodes values instance caller-options)
    instance))

(defn bind-resident-program
  "Bind one freshly validated EmittedParallelProgram to caller-owned resident LinkValues.

   `:buffer-bindings` maps compiler value IDs to whole, exact single-leaf LinkValue IDs;
   `:scalar-values` supplies only explicitly typed immutable scalar inputs. This function neither
   allocates nor treats node sources/roles as initialized. A containing LinkPlan/LinkModule must
   establish all ordered initialization and alias obligations."
  ([parallel-program request]
   (link/without-validation-context
    #(bind-resident-program-in-context parallel-program request nil)))
  ([parallel-program request caller-options]
   (link/without-validation-context
    #(bind-resident-program-in-context parallel-program request caller-options))))

(defn- completable-map-result?
  [parallel-program projections value-id]
  (let [producers (filterv #(some #{value-id} (:results %)) (:equations parallel-program))
        equation (first producers)
        operation (first (:operations equation))
        projection (when operation (emitted-program/operation-projection projections operation))
        algorithm (get-in projection [:boundary :algorithm])
        source-equations (when (soac/program-form? algorithm) (soac/equations algorithm))
        source-facts (when (soac/program-form? algorithm) (soac/facts algorithm))
        value (get-in parallel-program [:values value-id])]
    (and (= 1 (count producers) (count source-equations))
         (= value-id (get-in projection [:physical-results value-id]))
         (= 'map (soac/operation-kind (first source-equations)))
         (some #{value-id} (nth (first source-equations) 2))
         (empty? (dissoc (:attributes source-facts) :source-arithmetic))
         (contains? #{nil numerics/retained-source-arithmetic}
                    (get-in source-facts [:attributes :source-arithmetic]))
         (empty? (:effects source-facts)) (empty? (:aliases source-facts))
         (every? (fn [eq]
                   (let [ef (get-in source-facts [:equations (second eq)])]
                     (and (empty? (:effects ef)) (empty? (:aliases ef))
                          (empty? (:attributes ef))
                          (empty? (soac/result-storage algorithm (second eq)))))) source-equations)
         (= :tensor (:kind value)) (= {:kind :plain} (:representation value))
         (= :owned (:ownership value)) (nil? (:logical-layout value))
         (contains? #{nil :device} (:memory-space value))
         (nil? (:placement value)) (nil? (:sharding value))
         (empty? (:effects value)) (empty? (:attributes value))
         (vector? (:shape value)) (seq (:shape value))
         (every? #(and (integer? %) (pos? %)) (:shape value)))))

(defn- complete-resident-buffer-layout-in-context
  [parallel-program request caller-options]
  (let [{:keys [program retained-validation expected-buffers]}
        (resident-program-context! parallel-program request caller-options)
        {:keys [id target nodes values buffer-bindings scalar-values]} request
        supplied (set (keys buffer-bindings))
        missing (set/difference expected-buffers supplied)
        public (set/intersection expected-buffers (set (concat (:inputs program) (:outputs program))))]
    (when-not (and (set/subset? supplied expected-buffers) (set/subset? public supplied))
      (fail! :resident-layout-completion-boundary
             "resident completion requires all public buffers and no unknown bindings"
             {:expected expected-buffers :supplied supplied :public public}))
    (if (empty? missing)
      request
      (let [_ (when-not (every? #(completable-map-result? program (:projections retained-validation) %) missing)
                (fail! :resident-layout-completion-producer
                       "only private owned positive-static pure-map self-results may be completed"
                       {:missing missing}))
            tokens (into {} (map #(vector % (storage-id id :result %))) missing)
            existing-ids (into (set (keys nodes))
                               (concat (keys values) (map #(get-in % [:view :allocation :id]) (vals nodes))))
            new-ids (set (vals tokens))
            _ (when-not (and (= (count missing) (count new-ids))
                             (empty? (set/intersection existing-ids new-ids)))
                (fail! :resident-layout-completion-identity
                       "internal resident storage identities must be fresh across nodes, values and allocations"
                       {:identities new-ids :existing existing-ids}))
            initial (resident-single-leaf-storage nodes values (:values program) buffer-bindings)
            realized (reduce #(lower-equation-storage %1 id %2 (:values program) scalar-values
                                                      (:projections retained-validation))
                             initial (:equations program))
            new-storage (select-keys (:storage realized) new-ids)
            _ (when-not (and (= (:buffers realized) (merge buffer-bindings tokens))
                             (= (:storage initial) (apply dissoc (:storage realized) new-ids))
                             (= new-ids (set (keys new-storage)))
                             (empty? (:loop-scratch realized))
                             (every? #(and (nil? (:source %)) (= :unspecified (:initialization %))
                                           (nil? (:backing %)) (nil? (:result-view %)))
                                     (vals new-storage)))
                (fail! :resident-layout-completion-storage
                       "resident completion cannot rewrite supplied storage or introduce views or initializers"
                       {:missing missing :buffers (:buffers realized)}))
            derived (storage-link-layout realized scalar-values target #{} (vals tokens))]
        (-> request
            (update :nodes into (map (juxt :id identity) (:nodes derived)))
            (update :values into (map (juxt :id identity) (:values derived)))
            (assoc :buffer-bindings (:buffers realized)))))))

(defn complete-resident-buffer-layout
  "Complete only missing private, uniquely defined owned positive-static pure-map results.

   Takes the same closed storage request as bind-resident-program. All public buffer inputs and
   outputs and exact typed scalars must already be supplied. Reuses ordinary storage lowering;
   adds only fresh internal whole-buffer node/value DATA, never host content, initialization,
   driver allocations or a validation receipt. An already complete request is returned unchanged.
   The strict binder and a final LinkPlan/LinkModule must still prove storage and readiness."
  ([parallel-program request]
   (link/without-validation-context
    #(complete-resident-buffer-layout-in-context parallel-program request nil)))
  ([parallel-program request caller-options]
   (link/without-validation-context
    #(complete-resident-buffer-layout-in-context parallel-program request caller-options))))

(defn lower
  "Lower a specialized invocation under independent caller math intent, without driver allocation."
  ([materialized parallel-program target evaluate-host]
   (:plan (lower-for-request materialized parallel-program target evaluate-host
                            (fn [plan] {:plan plan}) nil nil)))
  ([materialized parallel-program target evaluate-host project]
   (lower-for-request materialized parallel-program target evaluate-host project nil nil))
  ([materialized parallel-program target evaluate-host project retained-validation]
   (lower-for-request materialized parallel-program target evaluate-host project retained-validation nil))
  ([materialized parallel-program target evaluate-host project retained-validation caller-options]
   (lower-for-request materialized parallel-program target evaluate-host project retained-validation caller-options)))

(defn ^:no-doc project-public-roles
  "Project semantic public roles onto one typed invocation candidate. This is a transformation,
   not a validation witness: the ordinary final LinkPlan proof must check the resulting boundary."
  [candidate roles]
  (let [bindings (get-in candidate [:attributes :public-buffer-bindings])
        compiler-bindings (get-in candidate [:attributes :compiler-buffer-bindings])]
    (when-not (and (map? bindings) (map? compiler-bindings) (map? roles)
                   (= 1 (count (:instances candidate)))
                   (set/subset? (set (keys roles)) (set (keys bindings))))
      (fail! :invocation-public-role-boundary "public roles require one exact typed invocation boundary" {}))
    (let [token-roles (reduce-kv (fn [result id role]
                                   (let [token (get bindings id)]
                                     (when (and (contains? result token) (not= role (get result token)))
                                       (fail! :invocation-public-role-alias
                                              "aliased public ports cannot have different roles"
                                              {:node token}))
                                     (assoc result token role))) {} roles)
          compiler-roles (into {} (keep (fn [[id token]]
                                         (when-let [role (get token-roles token)] [id role])))
                               compiler-bindings)]
      (-> (reduce-kv (fn [plan token role] (assoc-in plan [:nodes token :role] role))
                     candidate token-roles)
          (assoc-in [:instances 0 :roles] compiler-roles)
          (assoc-in [:attributes :public-buffer-roles] roles)))))

(defn ^:no-doc finish-ordered-realization
  "Construct and independently validate the ordinary whole-program boundary after source work.
   Cached host results are looked up, not recomputed. Zeroed source allocations remain conditional
   initialized inputs; the runtime must prove their actual completed initialization before export."
  [state caller-options]
  (when-not (= caller-options (:caller-options state))
    (fail! :invocation-ordered-request "ordered realization cannot change its caller math request" {}))
  (when-not (and (nil? (:pending state))
                 (= (:source-cursor state) (count (:source-order state)))
                 (or (empty? (:terminal-exports state)) (:terminal-observed? state)))
    (fail! :invocation-ordered-incomplete "source work has not completed" {}))
  (let [materialized (materialization/finish-materialization (:materialization state))
        initialized-tokens (into #{}
                                 (keep (fn [[token storage]]
                                         (when (= :zero (:initialization storage)) token)))
                                 (get-in state [:realized :storage]))
        evaluate-host (fn [equation _]
                        (let [step (get-in state [:host-steps (:id equation)])]
                          (when-not (= equation (:equation step))
                            (fail! :invocation-ordered-host-result
                                   "completed source work lacks this exact host definition"
                                   {:equation (:id equation)}))
                          (:results step)))
        entry-refresh (:entry-refresh state)
        captured (into #{} (keep (fn [[token role]] (when (not= :input role) token))) entry-refresh)
        project (fn [candidate]
                  {:plan (cond-> (update candidate :nodes
                                 (fn [nodes]
                                   (let [nodes (reduce (fn [nodes token]
                                                        (if (contains? nodes token)
                                                          (assoc-in nodes [token :source] nil)
                                                          nodes))
                                                       nodes captured)]
                                     (reduce (fn [nodes token]
                                               (when-not (contains? nodes token)
                                                 (fail! :invocation-ordered-storage
                                                        "source allocation is absent from final storage"
                                                        {:storage token}))
                                               (assoc-in nodes [token :role] :state))
                                             nodes initialized-tokens))))
                           entry-refresh
                           (project-public-roles
                            (into {} (keep (fn [[symbol token]]
                                             (when-let [role (get entry-refresh token)] [symbol role])))
                                  (get-in candidate [:attributes :public-buffer-bindings]))))})]
    (link/without-validation-context
     #(lower-in-context materialized (:program state) (:target state) evaluate-host project nil
                        caller-options state))))

(def ^:private resident-layout-keys
  #{:target :nodes :values :buffer-bindings :scalar-values :public-inputs})

(defn- resident-layout-projection
  [candidate caller-options]
  ;; The ordinary initialized candidate is the authority for initializer, storage and effect
  ;; correctness. Validate it before removing the caller arrays; the projected plan is then
  ;; independently validated by make-with-final-projection as a conditional input boundary.
  (if (nil? caller-options)
    (link/validate! candidate)
    (link/validate! candidate caller-options))
  (let [{:keys [nodes instances attributes]} candidate
        public-bindings (:public-buffer-bindings attributes)
        public-roles (:public-buffer-roles attributes)
        public-tokens (set (vals public-bindings))]
    (when-not (and (= 1 (count instances))
                   (link/program-link-instance? (first instances)))
      (fail! :resident-layout-instance
             "resident layout projection requires one ordinary emitted program instance"
             {:instances (mapv type instances)}))
    (when-not (and (map? public-bindings) (map? public-roles)
                   (= (set (keys public-bindings)) (set (keys public-roles)))
                   (every? #(= :input %) (vals public-roles)))
      (fail! :resident-layout-public-roles
             "resident layout projection admits only read-only public buffer inputs"
             {:bindings public-bindings :roles public-roles}))
    (when-not (= (count public-tokens) (count public-bindings))
      (fail! :resident-layout-public-alias
             "resident public inputs require distinct whole storage identities"
             {:bindings public-bindings}))
    (doseq [[node-id node] nodes]
      (cond
        (contains? public-tokens node-id)
        (when-not (and (= :state (:role node)) (some? (:source node)))
          (fail! :resident-layout-public-source
                 "a resident public input must originate in one checked host initializer"
                 {:node node-id :role (:role node) :source? (some? (:source node))}))

        (some? (:source node))
        (fail! :resident-layout-retained-source
               "resident layout projection cannot retain a non-public host initializer"
               {:node node-id :role (:role node)})))
    (when (or (seq (:host-outputs attributes))
              (not= 0 (:driver-allocations attributes)))
      (fail! :resident-layout-host-boundary
             "resident layout projection cannot retain host outputs or driver allocation"
             {:host-outputs (:host-outputs attributes)
              :driver-allocations (:driver-allocations attributes)}))
    {:plan (reduce (fn [plan node-id]
                     (-> plan
                         (assoc-in [:nodes node-id :role] :input)
                         (assoc-in [:nodes node-id :source] nil)))
                   candidate public-tokens)}))

(defn- ordered-resident-public-inputs
  [invocation-plan public-bindings]
  (let [inputs (into []
                     (keep (fn [{:keys [symbol]}]
                             (when-let [value (get public-bindings symbol)]
                               {:symbol symbol :value value})))
                     (:parameters invocation-plan))
        symbols (mapv :symbol inputs)]
    (when-not (and (= (count public-bindings) (count inputs) (count (distinct symbols)))
                   (= (set (keys public-bindings)) (set symbols)))
      (fail! :resident-layout-public-inputs
             "resident public inputs must exactly follow the invocation parameter boundary"
             {:parameters (mapv :symbol (:parameters invocation-plan))
              :bindings (set (keys public-bindings)) :projected symbols}))
    inputs))

(defn project-resident-layout
  "Project one materialized invocation to non-executable resident storage data.

   Ordinary invocation lowering remains the sole shape, type, initializer and physical-layout
   authority. Public read-only buffer sources are checked, then removed before final LinkPlan
   validation. The returned closed map is neither a LinkPlan nor proof: consumers must select its
   identities through bind-resident-program and validate a complete LinkPlan/LinkModule. It carries
   no host buffer initializer, executable instance, output capability, callback or validation
   receipt. AbstractValue metadata and typed immutable scalar values remain authoritative."
  ([materialized parallel-program target]
   (project-resident-layout materialized parallel-program target nil))
  ([materialized parallel-program target caller-options]
   (let [materialized (materialization/validate! materialized)
         invocation-plan (:plan materialized)
         refuse-host (fn [& _]
                       (fail! :resident-layout-host-boundary
                              "resident layout projection cannot evaluate host equations" {}))
         result (lower-for-request
                 materialized parallel-program target refuse-host
                 #(resident-layout-projection % caller-options) nil caller-options)
         plan (:plan result)
         instance (first (:instances plan))
         call (:call instance)
         attributes (:attributes plan)
         public-bindings (:public-buffer-bindings attributes)
         public-inputs (ordered-resident-public-inputs invocation-plan public-bindings)
         snapshot {:target target
                   :nodes (:nodes plan)
                   :values (:values plan)
                   :buffer-bindings (:compiler-buffer-bindings attributes)
                   :scalar-values (:scalar-values call)
                   :public-inputs public-inputs}]
     (when-not (= resident-layout-keys (set (keys snapshot)))
       (fail! :resident-layout-snapshot
              "resident layout snapshot must remain a closed non-executable value"
              {:keys (set (keys snapshot))}))
     (when (or (some (comp some? :source val) (:nodes snapshot))
               (some program-call/evaluated-host-equation? (:steps call)))
       (fail! :resident-layout-host-boundary
              "resident layout snapshot retained host execution state" {}))
     snapshot)))

(def ^:private resident-layout-clone-keys
  #{:template-value :template-compiler-value :template-abstract
    :compiler-value :abstract :value-id :node-id :allocation-id :role
    :project-ownership?})

(defn- resident-layout-storage
  [{:keys [nodes values]}]
  (into {}
        (map (fn [[value-id value]]
               (let [leaves (:leaves value)]
                 (when-not (= 1 (count leaves))
                   (fail! :resident-layout-clone-leaves
                          "resident layout cloning requires one whole physical leaf"
                          {:value value-id :leaves leaves}))
                 (let [node-id (get-in leaves [0 :node])
                       node (get nodes node-id)]
                   (when-not node
                     (fail! :resident-layout-clone-node
                            "resident layout cloning found an absent physical leaf"
                            {:value value-id :node node-id}))
                   [value-id {:shape (get-in node [:view :shape])}]))))
        values))

(defn ^:no-doc clone-resident-value-layout
  "Create one fresh standalone resident value from a validated layout's concrete geometry.

   This is construction, not evidence. The caller supplies the canonical source/target
   AbstractValues and compiler identities; this function independently realizes both shapes
   through the existing invocation dimension grammar, requires exact storage correspondence,
   and preserves every target facet except the proved concrete shape and the one explicit
   borrowed-to-owned public-input projection. A later resident binder and LinkPlan must validate
   the returned node/value in their complete program context."
  [layout request]
  (when-not (and (map? layout) (= resident-layout-keys (set (keys layout))))
    (fail! :resident-layout-clone-layout
           "resident value cloning requires one closed projected layout"
           {:keys (when (map? layout) (set (keys layout)))}))
  (when-not (and (map? request) (= resident-layout-clone-keys (set (keys request))))
    (fail! :resident-layout-clone-request
           "resident value cloning requires one closed construction request"
           {:keys (when (map? request) (set (keys request)))}))
  (let [{:keys [target nodes values buffer-bindings scalar-values]} layout
        {:keys [template-value template-compiler-value template-abstract compiler-value abstract
                value-id node-id allocation-id role project-ownership?]} request
        template-abstract (av/validate! template-abstract)
        abstract (av/validate! abstract)
        template (get values template-value)
        leaves (:leaves template)
        template-node-id (get-in leaves [0 :node])
        template-node (get nodes template-node-id)
        _ (when-not (and template template-node (= 1 (count leaves)))
            (fail! :resident-layout-clone-template
                   "resident value cloning requires one exact template value and leaf"
                   {:template template-value :leaves leaves :node template-node-id}))
        view (bview/validate-view! (:view template-node))
        allocation (:allocation view)
        storage (resident-layout-storage layout)
        template-shape (concrete-shape template-compiler-value template-abstract scalar-values
                                       buffer-bindings storage)
        target-shape (concrete-shape compiler-value abstract scalar-values
                                     buffer-bindings storage)
        allocation-users
        (filterv (fn [[_ node]]
                   (= [(:device allocation) (:id allocation)]
                      [(get-in node [:view :allocation :device])
                       (get-in node [:view :allocation :id])]))
                 nodes)
        node-users (filterv (fn [[_ value]]
                              (some #(= template-node-id (:node %)) (:leaves value)))
                            values)
        known-allocation-ids (into #{} (map #(get-in % [:view :allocation :id])) (vals nodes))
        existing-identities (into (set (keys nodes))
                                  (concat (keys values) known-allocation-ids))
        new-identities (set [value-id node-id allocation-id])
        projected (cond
                    project-ownership?
                    (do
                      (when-not (= :borrowed (:ownership abstract))
                        (fail! :resident-layout-clone-ownership
                               "only a canonical borrowed input may project to owned module storage"
                               {:ownership (:ownership abstract)}))
                      (assoc abstract :shape target-shape :ownership :owned))

                    :else (assoc abstract :shape target-shape))]
    (when-not (and (contains? #{:input :output} role)
                   (= project-ownership? (= :input role))
                   (or project-ownership? (not= :borrowed (:ownership abstract)))
                   (= 3 (count new-identities))
                   (empty? (set/intersection existing-identities new-identities))
                   (every? some? [value-id node-id allocation-id compiler-value
                                  template-compiler-value]))
      (fail! :resident-layout-clone-identity
             "resident value cloning requires fresh stable identities and a valid node role"
             {:value value-id :node node-id :allocation allocation-id :role role}))
    (when-not (and (= target (:device allocation))
                   (= :owned (:ownership allocation))
                   (= :device (:memory-space allocation))
                   (nil? (:source template-node))
                   (= 0 (:byte-offset view))
                   (pos? (:byte-length view))
                   (= (:byte-length view) (:byte-size allocation))
                   (= (:strides view) (bview/dense-strides (:shape view)))
                   (every? pos? (:shape view))
                   (= {:kind :dense} (:physical-layout template))
                   (every? (comp nil? :source val) nodes)
                   (= 1 (count allocation-users))
                   (= 1 (count node-users)))
      (fail! :resident-layout-clone-storage
             "resident value cloning requires one dense standalone owned allocation"
             {:template template-value :node template-node-id :view view
              :allocation-users (mapv first allocation-users)
              :value-users (mapv first node-users)}))
    (when-not (and (= template-shape (:shape view))
                   (= target-shape (:shape view))
                   (= (:shape template-abstract) (:shape abstract))
                   (= template-value (get buffer-bindings template-compiler-value))
                   (= (:abstract template) (assoc template-abstract :shape template-shape))
                   (= (:dtype (av/storage-contract abstract)) (:dtype view))
                   (av/storage-contract-compatible? template-abstract abstract)
                   (= (dissoc abstract :shape :ownership)
                      (dissoc projected :shape :ownership))
                   (or project-ownership? (= (:ownership abstract) (:ownership projected))))
      (fail! :resident-layout-clone-contract
             "canonical target and template contracts do not realize the same storage"
             {:template-shape template-shape :target-shape target-shape
              :view-shape (:shape view) :template template-abstract :target abstract}))
    (let [node (link/node {:id node-id :allocation-id allocation-id :dtype (:dtype view)
                           :shape (:shape view) :strides (:strides view)
                           :byte-size (:byte-size allocation) :device target
                           :memory-space (:memory-space allocation)
                           :ownership :owned :alignment (:alignment allocation)
                           :coherence (:coherence allocation) :role role})
          value (link/value {:id value-id :abstract (av/validate! projected)
                             :physical-layout (:physical-layout template)
                             :leaves [{:name (get-in leaves [0 :name]) :node node-id}]})]
      {:node node :value value})))

(defn- derive-certificate
  [plan effect-evidence]
  (let [attributes (:attributes plan)]
    (when-not (= :typed-invocation (:source attributes))
      (fail! :invocation-link-certificate-source
             "an invocation certificate requires a typed-invocation LinkPlan"
             {:plan (:id plan) :source (:source attributes)}))
    (when-not (every? link/program-link-instance? (:instances plan))
      (fail! :invocation-link-certificate-instances
             "an invocation certificate requires emitted program instances"
             {:plan (:id plan) :instances (mapv type (:instances plan))}))
    (->InvocationLinkCertificate
     :emitted-parallel-program-call :link-plan (:id plan) (:target plan)
     (mapv :id (:instances plan))
     (:public-buffer-bindings attributes)
     (:public-buffer-roles attributes)
     (:compiler-buffer-bindings attributes)
     (:semantic-outputs attributes)
     (:outputs plan) (:aliases plan)
     (:driver-allocations attributes) effect-evidence)))

(defn- verify-for-request!
  "Revalidate a typed invocation LinkPlan and independently rederive its composition witness."
  [lowering caller-options]
  (when-not (certified-link? lowering)
    (fail! :invocation-link-certificate-type
           "expected a CertifiedInvocationLink"
           {:actual (type lowering)}))
  (let [{:keys [plan effect-evidence]}
        (if (nil? caller-options)
          (link/validate-with-effect-evidence! (:plan lowering))
          (link/validate-with-effect-evidence! (:plan lowering) nil caller-options))
        expected (derive-certificate plan effect-evidence)]
    (when-not (certificate? (:certificate lowering))
      (fail! :invocation-link-certificate-type
             "typed invocation lowering requires an InvocationLinkCertificate"
             {:actual (type (:certificate lowering))}))
    (when-not (= expected (:certificate lowering))
      (fail! :invocation-link-certificate
             "typed invocation certificate does not match its LinkPlan"
             {:expected expected :actual (:certificate lowering)}))
    lowering))

(defn verify!
  "Independently reconstruct an invocation certificate under caller math intent."
  ([lowering] (verify-for-request! lowering nil))
  ([lowering caller-options] (verify-for-request! lowering caller-options)))

(defn- retention-position
  [equation-index equation phase]
  {:equation-index equation-index :equation equation :phase phase})

(defn- position-key
  [position]
  (cond
    (= :entry position) [Long/MIN_VALUE 0]
    (= :escape position) [Long/MAX_VALUE 0]
    :else [(:equation-index position) (case (:phase position) :read 0 :write 1)]))

(defn- later-position?
  [left right]
  (pos? (compare (position-key left) (position-key right))))

(defn- before-position?
  [left right]
  (neg? (compare (position-key left) (position-key right))))

(defn- value-retention-witness
  "Derive semantic value intervals over one verified invocation's physical storage.

   This is deliberately not a liveness or completion certificate.  It records equation-order
   read-before-write positions and rejects storage histories that cannot be justified from the
   existing complete-write effect evidence. Individual storage entries remain diagnostic when the
   aggregate status is unknown; any future safety decision must require the aggregate status."
  [plan certificate report emitted bindings definitions uses]
  (let [equations (:equations emitted)
        equation-index (into {} (map-indexed (fn [index equation] [(:id equation) index])) equations)
        inputs (set (:inputs emitted))
        semantic-outputs (set (:outputs emitted))
        output-nodes (set (:outputs plan))
        step-facts (get-in certificate [:effect-evidence :step-facts])
        storage-order (vec (distinct (vals bindings)))
        storage-nodes (fn [storage]
                        (mapv :node (get-in report [:values storage :leaves])))
        node-storage (into {} (map (fn [[node facts]] [node (:value facts)])) (:nodes report))
        allocation-nodes (reduce-kv (fn [groups node {:keys [allocation]}]
                                      (update groups allocation (fnil conj []) node))
                                    {} (:nodes report))
        semantic-values
        (into {}
              (keep (fn [[compiler-value storage]]
                      (let [definition (or (get definitions compiler-value)
                                           (when (contains? inputs compiler-value)
                                             {:kind :program-input}))]
                        (when definition
                          (let [definition-position
                                (if (= :program-input (:kind definition))
                                  :entry
                                  (retention-position (get equation-index (:id definition))
                                                      (:id definition) :write))
                                use-positions
                                (mapv (fn [equation]
                                        (retention-position (get equation-index equation)
                                                            equation :read))
                                      (get uses compiler-value []))
                                logical-last-use (reduce (fn [latest position]
                                                           (if (later-position? position latest)
                                                             position latest))
                                                         definition-position use-positions)
                                semantic-output? (contains? semantic-outputs compiler-value)]
                            [compiler-value
                             {:storage storage
                              :definition-position definition-position
                              :use-positions use-positions
                              :logical-last-use logical-last-use
                              :semantic-output? semantic-output?
                              :retained-through (if semantic-output? :escape logical-last-use)
                              :physical-storage-escaped?
                              (boolean (some output-nodes (storage-nodes storage)))}])))))
              bindings)
        versions-by-storage
        (reduce-kv (fn [groups compiler-value {:keys [storage]}]
                     (update groups storage (fnil conj []) compiler-value))
                   {} semantic-values)
        definition-storage-by-step
        (reduce-kv (fn [by-step _ {:keys [storage definition-position]}]
                     (if (map? definition-position)
                       (update by-step (:equation-index definition-position)
                               (fnil conj #{}) storage)
                       by-step))
                   {} semantic-values)
        write-events
        (reduce
         (fn [events {:keys [step phase facts complete-writes]}]
           (if-not (integer? step)
             events
             (let [by-node
                   (reduce (fn [nodes {:keys [node access symbol]}]
                             (-> nodes
                                 (update-in [node :access] kabi/merge-access access)
                                 (update-in [node :symbols] (fnil conj []) symbol)))
                           {} facts)]
               (reduce-kv
                (fn [events node {:keys [access symbols]}]
                  (if (contains? #{:write :read-write} access)
                    (update events (get node-storage node) (fnil conj [])
                            {:equation-index step :equation phase :node node :symbols symbols
                             :access access
                             :complete-write? (boolean (and complete-writes
                                                            (contains? complete-writes node)))})
                    events))
                events by-node))))
         {} step-facts)
        global-unknown
        (cond-> []
          (not= 1 (count (:instances plan)))
          (conj {:reason :multiple-program-instances :count (count (:instances plan))})

          (some emitted-loop/emitted-loop? (mapcat :operations equations))
          (conj {:reason :structured-replay}))
        storage-report
        (into {}
              (map
               (fn [storage]
                 (let [nodes (storage-nodes storage)
                       node-facts (mapv #(get-in report [:nodes %]) nodes)
                       versions (->> (get versions-by-storage storage [])
                                     (sort-by #(position-key
                                                (get-in semantic-values [% :definition-position])))
                                     vec)
                       events (get write-events storage [])
                       boundary-reasons
                       (cond-> #{}
                         (some #(contains? #{:input :constant :state :output} (:role %)) node-facts)
                         (conj :persistent-role)
                         (some :host-initializer? node-facts) (conj :host-initialized)
                         (some output-nodes nodes) (conj :physical-output))
                       reasons
                       (cond-> []
                         (not= 1 (count nodes))
                         (conj {:reason :multi-leaf-value :nodes nodes})

                         (some #(> (count (get allocation-nodes (:allocation %))) 1) node-facts)
                         (conj {:reason :multi-view-allocation :nodes nodes})

                         (some #(not (:complete-write? %)) events)
                         (conj {:reason :partial-write
                                :events (filterv #(not (:complete-write? %)) events)})

                         (some #(= :read-write (:access %)) events)
                         (conj {:reason :read-write-mutation
                                :events (filterv #(= :read-write (:access %)) events)})

                         (some (fn [{:keys [equation-index]}]
                                 (not (contains? (get definition-storage-by-step equation-index #{})
                                                 storage)))
                               events)
                         (conj {:reason :unattributed-mutation
                                :events (filterv
                                         (fn [{:keys [equation-index]}]
                                           (not (contains?
                                                 (get definition-storage-by-step equation-index #{})
                                                 storage)))
                                         events)}))
                       reasons
                       (reduce
                        (fn [reasons compiler-value]
                          (let [definition-position
                                (get-in semantic-values [compiler-value :definition-position])]
                            (if (and (map? definition-position)
                                     (not-any? #(and (= (:equation-index definition-position)
                                                        (:equation-index %))
                                                     (:complete-write? %))
                                               events))
                              (conj reasons {:reason :definition-without-complete-write
                                             :value compiler-value
                                             :position definition-position})
                              reasons)))
                        reasons versions)
                       reasons
                       (reduce
                        (fn [reasons [prior next]]
                          (let [prior-retention
                                (if (get-in semantic-values [prior :semantic-output?])
                                  :escape
                                  (get-in semantic-values [prior :logical-last-use]))
                                next-definition
                                (get-in semantic-values [next :definition-position])]
                            (if-not (before-position? prior-retention next-definition)
                              (conj reasons {:reason (if (= :escape prior-retention)
                                                       :escaped-old-version
                                                       :overlapping-value-versions)
                                             :prior prior :next next
                                             :prior-retained-through prior-retention
                                             :next-definition next-definition})
                              reasons)))
                        reasons (partition 2 1 versions))
                       reasons
                       (if (and (some output-nodes nodes) (< 1 (count versions))
                                (not-any? #(get-in semantic-values [% :semantic-output?]) versions))
                         (conj reasons {:reason :ambiguous-physical-escape :versions versions})
                         reasons)
                       status (cond (seq reasons) :unknown
                                    (seq boundary-reasons) :persistent-boundary
                                    :else :witnessed)]
                   [storage (cond-> {:status status :versions versions :nodes nodes}
                              (seq boundary-reasons) (assoc :boundary-reasons boundary-reasons)
                              (seq reasons) (assoc :unknown reasons))])))
              storage-order)
        unknown (into global-unknown
                      (mapcat (fn [[storage {:keys [unknown]}]]
                                (map #(assoc % :storage storage) unknown)))
                      storage-report)]
    {:status (if (seq unknown) :unknown :witnessed)
     :ordering :semantic-equation-read-before-write
     :completion :unproven
     :values semantic-values
     :storage storage-report
     :unknown (vec unknown)}))

(defn- memory-witness-for-request
  "Join a verified typed invocation's compiler storage identities to its LinkPlan memory facts.

   Compiler values may share a storage identity; physical destination names with no semantic
   SSA definition are marked :storage-only and are not treated as versions. Straight-line semantic
   retention is derived in equation read-before-write order and checked against complete-write
   effect evidence. Unsupported storage histories remain explicit unknowns. A future safety
   consumer must require the aggregate :value-versions status, not cherry-pick a storage entry.
   The returned report makes no selected-order, release, reuse, or completion decision."
  [lowering caller-options]
  (let [{:keys [plan certificate]} (if (nil? caller-options)
                                    (verify! lowering) (verify! lowering caller-options))
        report (if (nil? caller-options)
                 (link/memory-report plan) (link/memory-report plan caller-options))
        bindings (:compiler-buffer-bindings certificate)
        ;; verify! already revalidates the bound EmittedParallelProgramCall through LinkPlan.
        call (get-in plan [:instances 0 :call])
        emitted (:program call)
        actual-bindings (set (if (nil? caller-options)
                              (program-call/buffer-bindings call)
                              (program-call/buffer-bindings call caller-options)))
        reported-bindings (set bindings)
        definitions (into {}
                          (mapcat (fn [{:keys [id results]}]
                                    (map (fn [value] [value {:kind :equation :id id}]) results)))
                          (:equations emitted))
        uses (reduce (fn [sites {:keys [id operands]}]
                       (reduce (fn [sites value]
                                 (update sites value (fnil conj []) id))
                               sites (distinct operands)))
                     {} (:equations emitted))
        inputs (set (:inputs emitted))
        semantic-outputs (set (:outputs emitted))
        version-witness
        (cond-> (value-retention-witness plan certificate report emitted bindings definitions uses)
          (not= actual-bindings reported-bindings)
          (assoc :status :unknown)
          (not= actual-bindings reported-bindings)
          (update :unknown conj
                  {:reason :incomplete-compiler-buffer-bindings
                   :missing (set/difference actual-bindings reported-bindings)
                   :unexpected (set/difference reported-bindings actual-bindings)}))]
    (doseq [[compiler-value storage-id] bindings]
      (when-not (contains? (:values report) storage-id)
        (fail! :invocation-memory-binding
               "a compiler value has no validated logical storage in the LinkPlan"
               {:compiler-value compiler-value :storage storage-id :plan (:id plan)}))
      (when-not (contains? (:values emitted) compiler-value)
        (fail! :invocation-memory-value
               "a linked buffer has no typed value in its emitted program"
               {:compiler-value compiler-value :storage storage-id :plan (:id plan)})))
    {:source-dialect (:source-dialect certificate)
     :target-dialect (:target-dialect certificate)
     :plan (:id plan)
     :compiler-buffer-bindings bindings
     :compiler-values
     (into {}
           (map (fn [[compiler-value storage-id]]
                  [compiler-value
                   {:abstract (get-in emitted [:values compiler-value])
                    :storage storage-id
                    :definition (or (get definitions compiler-value)
                                    (when (contains? inputs compiler-value)
                                      {:kind :program-input})
                                    {:kind :storage-only})
                    :uses (get uses compiler-value [])
                    :semantic-output? (contains? semantic-outputs compiler-value)
                    :public-output? (contains? (get-in version-witness
                                                       [:storage storage-id :boundary-reasons] #{})
                                               :physical-output)
                    :retention (get-in version-witness [:values compiler-value])}]))
           bindings)
     :public-buffer-bindings (:public-buffer-bindings certificate)
     :semantic-outputs (:semantic-outputs certificate)
     :memory report
     :value-versions (dissoc version-witness :values)}))

(defn memory-witness
  "Report verified invocation storage/value retention under independent math intent, not completion."
  ([lowering] (memory-witness-for-request lowering nil))
  ([lowering caller-options] (memory-witness-for-request lowering caller-options)))

(defn- certify-for-request
  "Wrap a validated equation-first invocation LinkPlan in a checkable composition witness.

   An explicit output vector declares every escaped physical node (including taps and donations)
   before validation. It may extend/reorder, but never drop, the original semantic boundary.
   This retains physical storage; it does not snapshot earlier SSA versions sharing that storage."
  [plan outputs caller-options]
  (when (some? outputs)
    (when-not (vector? outputs)
      (fail! :invocation-link-output-boundary "escaped invocation outputs must be an ordered vector"
             {:outputs outputs}))
    (let [missing (set/difference (set (:outputs plan)) (set outputs))]
      (when (seq missing)
        (fail! :invocation-link-output-boundary
               "escaped invocation outputs must retain the semantic output boundary"
               {:missing missing :outputs outputs}))))
  (let [plan (if (some? outputs) (assoc plan :outputs outputs) plan)
        {:keys [plan effect-evidence]} (if (nil? caller-options)
                                        (link/validate-with-effect-evidence! plan)
                                        (link/validate-with-effect-evidence! plan nil caller-options))]
     ;; Construction and certificate derivation share the same validated immutable plan. External
     ;; boundaries retain `verify!` for independent re-derivation; repeating it here proves no new
     ;; fact and made every equation-first preparation pay for the certificate twice.
     (->CertifiedInvocationLink plan (derive-certificate plan effect-evidence))))

(defn certify
  "Certify under independent caller intent. An output vector may extend, never drop, escaped outputs."
  ([plan] (certify-for-request plan nil nil))
  ([plan outputs]
   (when-not (vector? outputs)
     (fail! :invocation-link-output-boundary
            "escaped invocation outputs must be an ordered vector"
            {:outputs outputs}))
   (certify-for-request plan outputs nil))
  ([plan outputs caller-options] (certify-for-request plan outputs caller-options)))

(defn- certify-final-projection-for-request
  "Certify the exact final LinkPlan just validated during construction. The retained effect
   witness is valid only for that same in-process plan; arbitrary callers use `certify`."
  [{:keys [plan effect-evidence]} caller-options]
  (when-not (if (nil? caller-options)
              (link/retained-effect-evidence? plan effect-evidence)
              (link/retained-effect-evidence? plan effect-evidence caller-options))
    (fail! :invocation-link-effect-evidence
           "final projection requires evidence sealed to its exact validated plan"
           {:plan (:id plan)}))
  (->CertifiedInvocationLink plan (derive-certificate plan effect-evidence)))

(defn ^:no-doc certify-final-projection
  "Certify only exact final-plan evidence checked against this independent math request."
  ([projected] (certify-final-projection-for-request projected nil))
  ([projected caller-options] (certify-final-projection-for-request projected caller-options)))
