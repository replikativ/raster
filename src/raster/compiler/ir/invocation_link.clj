(ns raster.compiler.ir.invocation-link
  "Pure lowering from a specialized public invocation to an equation-first LinkPlan.

   MaterializedInvocation owns public argument/scalar semantics. EmittedParallelProgram owns the
   numerical program. This namespace realizes their shared storage boundary exactly once: caller
   arrays become initialized LinkNodes, physical equation results and loop carry rotations become
   owned nodes, and the emitted call enters LinkPlan as a ProgramLinkInstance. No driver object,
   kernel name, or legacy resident descriptor participates."
  (:require [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :invocation-link))))

(defn- typed-scalar?
  [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn- scalar-number
  [scalars id]
  (let [scalar (get scalars id)]
    (when-not (and (typed-scalar? scalar) (integer? (:value scalar)))
      (fail! :invocation-link-shape-scalar
             "program storage shape requires an integral typed scalar"
             {:value id :scalar scalar :available (set (keys scalars))}))
    (long (:value scalar))))

(defn- concrete-shape
  [abstract scalars buffers storage]
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
             (scalar-number scalars dimension)
             (and (seq? dimension) (= 'extent (first dimension)) (= 2 (count dimension)))
             (let [source-id (get buffers (second dimension))
                   source (get storage source-id)]
               (when-not source
                 (fail! :invocation-link-shape-extent
                        "program extent projection names unavailable resident storage"
                        {:dimension dimension :buffers (set (keys buffers))}))
               (first (:shape source)))
             :else
             (fail! :invocation-link-shape-expression
                    "program storage shape is not a canonical integer/value/extent expression"
                    {:dimension dimension :shape (:shape abstract)}))]
       (when (neg? resolved)
         (fail! :invocation-link-shape-negative
                "program storage dimensions must be non-negative"
                {:dimension dimension :resolved resolved}))
       resolved))
   (:shape abstract)))

(defn- physical-output-map
  [algorithm]
  (let [facts (soac/facts algorithm)]
    (into {}
          (mapcat (fn [equation]
                    (map vector (nth equation 2) (soac/physical-results facts equation))))
          (soac/equations algorithm))))

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
          shape (concrete-shape abstract scalars (:buffers state) (:storage state))]
      [(-> state
           (assoc-in [:buffers compiler-value] token)
           (add-storage token compiler-value abstract shape nil :unspecified))
       token])))

(defn- add-materialized-inputs
  [state materialized program-values]
  (reduce-kv
   (fn [state compiler-value buffer]
     (let [token (:id buffer)
           expected (get program-values compiler-value)]
       (when-not expected
         (fail! :invocation-link-program-input
                "materialized buffer names a value absent from the emitted program"
                {:compiler-value compiler-value}))
       (when (= :zero (:initialization buffer))
         (fail! :invocation-link-zero-initializer
                "zero-initialized program storage requires an explicit initializer schedule"
                {:compiler-value compiler-value :storage token :shape (:shape buffer)}))
       (-> state
           (assoc-in [:buffers compiler-value] token)
           (add-storage token compiler-value expected (:shape buffer)
                        (:source buffer) (:initialization buffer)))))
   state (:program-buffers materialized)))

(defn- lower-loop-storage
  [state invocation-id equation scalars]
  (let [emitted (emitted-loop/validate! (first (:operations equation)))
        algorithm (-> emitted :schedule :algorithm)
        body-storage (physical-output-map (control/body algorithm))]
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
                 shape (concrete-shape output-abstract scalars (:buffers state) (:storage state))]
             (-> state
                 (assoc-in [:loop-scratch output] scratch-token)
                 (add-storage scratch-token output output-abstract shape nil :unspecified))))))
     state (control/carried algorithm))))

(defn- lower-equation-storage
  [state invocation-id equation program-values scalars]
  (cond
    (true? (get-in equation [:attributes :host-only])) state

    (emitted-loop/emitted-loop? (first (:operations equation)))
    (lower-loop-storage state invocation-id equation scalars)

    :else
    (let [emitted (emitted-equation/validate! (first (:operations equation)))
          physical (physical-output-map (:algorithm emitted))]
      (reduce
       (fn [state result]
         (let [physical-id (get physical result)
               abstract (or (get program-values physical-id) (get program-values result))
               [state token]
               (allocate-result state invocation-id physical-id abstract scalars :result)]
           (-> state
               (assoc-in [:buffers result] token)
               (update-in [:storage token :compiler-values] conj result))))
       state (:results equation)))))

(defn lower
  "Lower one specialized invocation and matching emitted program into a validated LinkPlan.

   `evaluate-host` is passed unchanged to EmittedParallelProgramCall for closed, effect-free host
   scalar equations. The returned plan is allocation-free; runtime contact starts only in
   raster.gpu.link/instantiate!."
  [materialized parallel-program target evaluate-host]
  (let [materialized (materialization/validate! materialized)
        parallel-program (emitted-program/validate! parallel-program)
        invocation-plan (:plan materialized)
        invocation-id (:id invocation-plan)
        _ (when-not (= (set (:program-inputs invocation-plan))
                       (set (:inputs parallel-program)))
            (fail! :invocation-link-program-boundary
                   "materialized invocation and emitted program have different input boundaries"
                   {:invocation-inputs (:program-inputs invocation-plan)
                    :program-inputs (:inputs parallel-program)}))
        scalars (:program-scalars materialized)
        initial (add-materialized-inputs {:buffers {} :loop-scratch {} :storage {}}
                                         materialized (:values parallel-program))
        realized (reduce #(lower-equation-storage %1 invocation-id %2
                                                  (:values parallel-program) scalars)
                         initial (:equations parallel-program))
        call (program-call/make parallel-program (:buffers realized) scalars
                                (:loop-scratch realized) evaluate-host)
        resident-outputs (into [] (remove (comp typed-scalar? val)) (:outputs call))
        output-tokens (set (map val resident-outputs))
        storage (:storage realized)
        nodes
        (mapv (fn [[token {:keys [shape source abstract]}]]
                (link/node {:id token :dtype (:dtype abstract) :shape shape :device target
                            :role (cond
                                    (and source (contains? output-tokens token)) :state
                                    source :state
                                    (contains? output-tokens token) :output
                                    :else :internal)
                            :source source}))
              storage)
        values
        (mapv (fn [[token {:keys [abstract]}]]
                (link/value {:id token :abstract abstract
                             :leaves [{:name :value :node token}]}))
              storage)
        instance (link/program-instance
                  {:id [invocation-id :emitted-program] :call call
                   :attributes {:source :typed-invocation}})]
    (link/make
     {:id [invocation-id :link-plan]
      :target target
      :nodes nodes
      :values values
      :instances [instance]
      :outputs (mapv val resident-outputs)
      :attributes {:source :typed-invocation
                   :invocation-plan invocation-id
                   :host-outputs (into {} (filter (comp typed-scalar? val)) (:outputs call))
                   :driver-allocations 0}})))
