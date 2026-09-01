(ns raster.compiler.ir.structured-loop-call
  "Pure host-repetition binding for an emitted structured-loop iteration graph.

   The call owns no driver handles. It maps outer values to the emitted graph boundary and plans
   double-buffered carry rotation. A runtime may bind/replay each returned iteration through its
   ordinary KernelGraph machinery."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.ir.soac-dialect :as soac]))

(defrecord StructuredLoopCall
           [schedule graph trip-count buffers scalars scratch outputs attributes])

(defn structured-loop-call?
  [value]
  (and value
       (= "raster.compiler.ir.structured_loop_call.StructuredLoopCall"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :structured-loop-call))))

(defn- typed-scalar?
  [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn- same-dtype?
  [left right]
  (= (dtype/canon left) (dtype/canon right)))

(defn- external-buffer-ids
  [emitted]
  (set (map :id (concat (:inputs emitted) (:outputs emitted)))))

(defn- scalar-slots
  [emitted]
  (into {}
        (keep (fn [[slot argument]]
                (when (= :scalar (:kind slot)) [argument slot])))
        (map vector (:abi emitted) (:arguments emitted))))

(defn- physical-results
  [algorithm]
  (let [body (control/body algorithm)
        facts (soac/facts body)]
    (into {}
          (mapcat (fn [equation]
                    (map vector (nth equation 2) (soac/physical-results facts equation))))
          (soac/equations body))))

(defn- resolve-trip-count
  [algorithm scalars]
  (let [trip-count (second (control/loop-index algorithm))
        resolved (if (integer? trip-count)
                   trip-count
                   (let [value (get scalars trip-count)]
                     (when-not (typed-scalar? value)
                       (fail! :structured-loop-trip-count
                              "symbolic loop trip count requires a typed scalar"
                              {:value trip-count :scalar value}))
                     (let [expected (get-in (control/outer-values algorithm)
                                            [trip-count :dtype])]
                       (when-not (same-dtype? expected (:type value))
                         (fail! :structured-loop-trip-count-type
                                "symbolic trip-count binding differs from its retained dtype"
                                {:value trip-count :expected expected :actual (:type value)})))
                     (:value value)))]
    (when-not (integer? resolved)
      (fail! :structured-loop-trip-count "loop trip count must resolve to an integer"
             {:resolved resolved}))
    (let [semantics (get-in (control/facts algorithm)
                            [:attributes :trip-count-semantics])]
      (cond
        (not (neg? resolved)) (long resolved)
        (= :clamp-nonnegative semantics) 0
        :else
        (fail! :structured-loop-trip-count "loop trip count must resolve non-negative"
               {:resolved resolved :semantics semantics})))))

(defn- scalar-binding
  [slots id expected-dtype value]
  (let [slot (get slots id)]
    (when-not slot
      (fail! :structured-loop-scalar-interface
             "structured loop scalar is absent from the emitted graph interface" {:value id}))
    (when-not (typed-scalar? value)
      (fail! :structured-loop-scalar "structured loop requires an explicitly typed scalar"
             {:value id :scalar value}))
    (when-not (and (same-dtype? expected-dtype (:kernel-dtype slot))
                   (same-dtype? expected-dtype (:type value)))
      (fail! :structured-loop-scalar-type
             "structured loop scalar, retained AbstractValue, and emitted ABI dtypes differ"
             {:value id :expected expected-dtype
              :emitted (:kernel-dtype slot) :actual (:type value)}))
    value))

(defn- require-buffer!
  [buffers id role]
  (let [value (get buffers id ::missing)]
    (when (or (= ::missing value) (nil? value))
      (fail! :structured-loop-buffer "structured loop buffer binding is missing"
             {:value id :role role}))
    value))

(defn validate!
  [call]
  (when-not (structured-loop-call? call)
    (fail! :structured-loop-call-type "expected a StructuredLoopCall" {:actual (type call)}))
  (let [{scheduled :schedule emitted :graph trip-count :trip-count
         buffers :buffers scalars :scalars scratch :scratch outputs :outputs attributes :attributes}
        call
        scheduled (schedule/validate! scheduled)
        emitted (graph/validate! emitted)]
    (when-not (every? (comp artifact/kernel-artifact? :operation) (:nodes emitted))
      (fail! :structured-loop-emitted-graph
             "structured loop call requires a fully emitted KernelGraph" {}))
    (when-not (and (graph/dataflow-equivalent? (:graph scheduled) emitted)
                   (every? true?
                           (map (fn [scheduled-node emitted-node]
                                  (= (:operation scheduled-node)
                                     (get-in emitted-node
                                             [:operation :provenance
                                              :scheduled-operation])))
                                (:nodes (:graph scheduled)) (:nodes emitted))))
      (fail! :structured-loop-emitted-graph
             "emitted graph dataflow or operation certificates differ from the scheduled iteration"
             {}))
    (when-not (and (integer? trip-count) (not (neg? trip-count)))
      (fail! :structured-loop-trip-count "resolved trip count must be non-negative"
             {:trip-count trip-count}))
    (doseq [[field value] [[:buffers buffers] [:scalars scalars] [:scratch scratch]
                           [:outputs outputs] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :structured-loop-call-field "structured loop call fields must be maps"
               {:field field :value value})))
    call))

(defn make
  "Bind outer logical values for target-neutral host repetition.

   `buffers` maps outer invariant/initial/output IDs to resident keys or views. `scalars` maps
   outer scalar IDs to typed values. `scratch` maps each carry output ID to its alternate buffer;
   it is required for an out-of-place carry only when the resolved trip count exceeds one."
  [scheduled emitted buffers scalars scratch]
  (let [scheduled (schedule/validate! scheduled)
        algorithm (:algorithm scheduled)
        emitted (graph/validate! emitted)
        graph-buffers (external-buffer-ids emitted)
        slots (scalar-slots emitted)
        inner-values (:values (soac/facts (control/body algorithm)))
        trip-count (resolve-trip-count algorithm scalars)
        iteration (first (control/loop-index algorithm))
        invariant-buffer-bindings
        (into {}
              (keep (fn [{:keys [outer parameter]}]
                      (when (contains? graph-buffers parameter)
                        [parameter (require-buffer! buffers outer :invariant)])))
              (control/invariants algorithm))
        invariant-scalar-bindings
        (into {}
              (keep (fn [{:keys [outer parameter]}]
                      (when (contains? slots parameter)
                        [parameter (scalar-binding slots parameter
                                                   (get-in inner-values [parameter :dtype])
                                                   (get scalars outer))])))
              (control/invariants algorithm))
        result-storage (physical-results algorithm)
        carry-plans
        (mapv (fn [{:keys [initial parameter result output]}]
                (let [physical-result (get result-storage result)
                      initial-buffer (require-buffer! buffers initial :carry-initial)
                      output-buffer (require-buffer! buffers output :carry-output)
                      in-place? (= parameter physical-result)
                      alternate (get scratch output)]
                  (when-not (and (contains? graph-buffers parameter)
                                 (contains? graph-buffers physical-result))
                    (fail! :structured-loop-carry-interface
                           "loop carry is absent from the emitted graph buffer interface"
                           {:parameter parameter :result physical-result}))
                  (if in-place?
                    (when-not (= initial-buffer output-buffer)
                      (fail! :structured-loop-in-place-carry
                             "an in-place carry requires one shared initial/output binding"
                             {:initial initial :output output}))
                    (do
                      (when (= initial-buffer output-buffer)
                        (fail! :structured-loop-carry-alias
                               "an out-of-place carry requires distinct initial/output buffers"
                               {:initial initial :output output}))
                      (when (and (> trip-count 1) (nil? alternate))
                        (fail! :structured-loop-carry-scratch
                               "multi-step out-of-place carry requires an alternate buffer"
                               {:output output :trip-count trip-count}))
                      (when (and alternate
                                 (or (= alternate initial-buffer) (= alternate output-buffer)))
                        (fail! :structured-loop-carry-scratch-alias
                               "carry scratch must differ from initial and output buffers"
                               {:output output}))))
                  {:initial-id initial :output-id output
                   :parameter parameter :result physical-result
                   :initial initial-buffer :output output-buffer
                   :alternate alternate :in-place? in-place?}))
              (control/carried algorithm))
        iteration-slot (get slots iteration)
        _ (when (and iteration-slot
                     (not (same-dtype? (get-in inner-values [iteration :dtype])
                                       (:kernel-dtype iteration-slot))))
            (fail! :structured-loop-index-type
                   "emitted induction ABI differs from its retained AbstractValue dtype"
                   {:iteration iteration
                    :expected (get-in inner-values [iteration :dtype])
                    :emitted (:kernel-dtype iteration-slot)}))
        _ (doseq [scalar-id (keys slots)]
            (when-not (or (= scalar-id iteration)
                          (contains? invariant-scalar-bindings scalar-id))
              (fail! :structured-loop-scalar-interface
                     "emitted iteration graph scalar is not a loop invariant or induction value"
                     {:value scalar-id})))
        output-bindings (into {}
                              (map (fn [{:keys [initial-id output-id initial output]}]
                                     [output-id (if (zero? trip-count) initial output)]))
                              carry-plans)
        call (->StructuredLoopCall
              scheduled emitted trip-count
              {:invariants invariant-buffer-bindings :carries carry-plans}
              {:invariants invariant-scalar-bindings
               :iteration (when iteration-slot
                            {:id iteration :type (:kernel-dtype iteration-slot)})}
              scratch output-bindings {:execution :host-repetition})]
    (validate! call)))

(defn iteration-binding
  "Return the ordinary KernelGraph buffer/scalar bindings for iteration `index`."
  [call index]
  (let [call (validate! call)
        trip-count (:trip-count call)]
    (when-not (and (integer? index) (<= 0 index) (< index trip-count))
      (fail! :structured-loop-iteration "iteration index is outside the loop trip count"
             {:index index :trip-count trip-count}))
    (let [last-index (dec trip-count)
          carry-bindings
          (into {}
                (mapcat
                 (fn [{:keys [parameter result initial output alternate in-place?]}]
                   (if in-place?
                     [[parameter initial]]
                     (let [first-destination (if (odd? trip-count) output alternate)
                           destination (if (even? index)
                                         first-destination
                                         (if (= first-destination output) alternate output))
                           source (if (zero? index)
                                    initial
                                    (if (= destination output) alternate output))]
                       [[parameter source] [result destination]])))
                 (get-in call [:buffers :carries])))
          iteration-scalar (get-in call [:scalars :iteration])]
      {:index index
       :last? (= index last-index)
       :buffers (merge (get-in call [:buffers :invariants]) carry-bindings)
       :scalar-values
       (cond-> (get-in call [:scalars :invariants])
         iteration-scalar
         (assoc (:id iteration-scalar)
                {:type (:type iteration-scalar) :value index}))})))
