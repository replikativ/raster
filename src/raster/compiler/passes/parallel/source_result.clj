(ns raster.compiler.passes.parallel.source-result
  "Prove complete source returns against the existing typed producer and storage authorities.

   Shared by flat and structured routes; this namespace grants no representation demand."
  (:require [clojure.set :as set]
            [raster.compiler.core.callee-identity :as callee-identity]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.device-scalar-schedule :as device-scalar]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :structured-control-route))))

(defn ^:no-doc ordered-distinct
  [values]
  (vec (distinct values)))

(defn ^:no-doc loop-boundary?
  [equation algorithm]
  (and (control/loop-program? algorithm)
       (= algorithm (control/validate! algorithm))
       (= (:operands equation) (ordered-distinct (control/outer-operands algorithm)))
       (= (:results equation) (control/outer-results algorithm))))

(defn ^:no-doc soac-boundary?
  [equation algorithm]
  (and (soac/program-form? algorithm)
       (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))


(defn ^:no-doc typed-algorithm-boundary?
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

(defn ^:no-doc scheduled-algorithm-boundary?
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

(defn ^:no-doc valid-scalar-cast-value?
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

(defn ^:no-doc canonicalize-value-dtype
  "Normalize only a known keyword dtype for producer/outer comparison; retain every other facet."
  [value]
  (if (and (av/abstract-value? value) (keyword? (:dtype value))
           (dtype/known? (:dtype value)))
    (update value :dtype dtype/canon)
    value))

(defn ^:no-doc certified-result-storage
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

(defn ^:no-doc validated-produced-value
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

(defn ^:no-doc certified-source-storage
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

(defn ^:no-doc source-result-projection
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
        result-destinations (into {} (map (fn [[result storage]]
                                             [result (:destination storage)]))
                                  storage-entries)
        buffer-output (fn [result]
                        (let [destination (get result-destinations result)]
                          {:kind :buffer :outputs [destination]
                           :output-labels [result]
                           :source-references #{destination}}))
        symbol-output (fn [symbol]
                        (let [symbol (resolve-alias symbol)]
                          (cond
                            (contains? effect-bindings symbol) {:kind :unit :outputs []}
                            (contains? buffer-host-bindings symbol)
                            (buffer-output (get buffer-host-bindings symbol))
                            (contains? outputs symbol)
                            (let [kind (cond
                                         (contains? stored-result-kinds symbol)
                                         (get stored-result-kinds symbol)
                                         (contains? stored-results symbol) nil
                                         (contains? array-types symbol) :buffer
                                         :else :value)]
                              (when kind
                                (if (contains? stored-result-kinds symbol)
                                  (buffer-output symbol)
                                  {:kind kind :outputs [symbol]})))
                            (contains? outputs (:output (get destinations symbol)))
                            (buffer-output (:output (get destinations symbol)))
                            :else nil)))]
    (letfn [(project [expression]
              (cond
                (symbol? expression)
                (symbol-output expression)

                (vector? expression)
                (loop [remaining expression projected [] labels [] references #{}]
                  (if (seq remaining)
                    (when-let [projection (project (first remaining))]
                      (recur (next remaining) (into projected (:outputs projection))
                             (into labels (or (:output-labels projection) (:outputs projection)))
                             (into references (:source-references projection))))
                    {:kind :tuple :outputs projected :output-labels labels
                     :source-references references}))

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

(defn ^:no-doc validate-source-result!
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
    (let [outputs (if (= 'void return-tag) [] (vec (distinct (:outputs projection))))
          ;; Nominal public keys remain independent of the physical view. These
          ;; labels grant no storage or execution authority and are rederived.
          labels (reduce (fn [labels [output label]]
                           (if (contains? labels output) labels (assoc labels output label)))
                         {} (map vector (:outputs projection)
                                 (or (:output-labels projection) (:outputs projection))))
          labels (into {} (filter (fn [[output label]] (not= output label)))
                       (select-keys labels outputs))
          available (into (set (:inputs parallel-program))
                          (mapcat :results (:equations parallel-program)))
          references (set/intersection (set outputs) (set (:source-references projection)))
          boundary-references (set/difference references available)
          parameters (set (or (:public-parameters options) (:active-params options)))
          source-pairs (vec (partition 2 (second source)))
          allocations (get-in parallel-program [:attributes :allocations])]
      (doseq [reference boundary-references]
        (let [bindings (filterv #(= reference (first %)) source-pairs)
              allocation (filterv #(= reference (:destination %)) allocations)
              value (get-in parallel-program [:values reference])]
          (when-not
           (and (= :tensor (:kind value)) (seq (:shape value))
                (or (and (contains? parameters reference) (empty? bindings))
                    (and (= 1 (count bindings))
                         (or (invocation-source/reference-identity-source
                              reference (second (first bindings)))
                             (and (= 1 (count allocation))
                                  (some? (:source-expression (first allocation)))
                                  (semantic/retained-source-equivalent?
                                   (:source-expression (first allocation))
                                   (second (first bindings))))))))
            (fail! :structured-control-source-reference
                   "a whole-buffer return requires a source parameter, reference alias or retained allocation"
                   {:reference reference}))))
      ;; A boundary reference is not a numerical operand/read. The ordinary
      ;; invocation constructor must still bind it, and LinkPlan independently
      ;; checks the untouched portion's initialization before publication.
      (assoc parallel-program :outputs outputs
             :inputs (into (:inputs parallel-program)
                           (filter boundary-references outputs))
             :attributes (cond-> (dissoc (:attributes parallel-program) :source-output-labels)
                           (seq labels) (assoc :source-output-labels labels))))))
