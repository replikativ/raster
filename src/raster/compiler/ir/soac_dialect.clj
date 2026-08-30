(ns raster.compiler.ir.soac-dialect
  "Typed, functional S-expression view of Raster's SOAC middle end.

   The expression spine is deliberately small and pattern-friendly.  Stable value/equation
   identity and compiler facts live in the explicit program envelope, never in Clojure metadata.
   ParallelProgram equations retain this verified functional program as their semantic algorithm;
   scheduled SegOps remain a separate ordered field.

   Canonical forms:

     (soac-program facts
       [(= equation-id [result ...]
           (scalar {:dtypes [:long ...]} [capture ...]
             (lambda [capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [result-expr ...]))))
        (= equation-id [result ...]
           (map {:index i :extent n} [array ...] [capture ...]
             (lambda [element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [result-expr ...]))))
        (= equation-id [result ...]
           (scatter {:index i :extent n :conflict :unique} [array ...] [capture ...]
             (lambda [element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [(write destination-index predicate value) ...]))))
        (= equation-id [result ...]
           (reduce {:index i :extent n
                    :accumulators [acc ...] :identities [zero ...]
                    :dtypes [:float ...] :algebra [{} ...]}
             [array ...] [capture ...]
             (lambda [acc ... element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [step-result ...]))))
        (= equation-id [result]
           (scan {:mode :inclusive :index i :extent n
                  :accumulators [acc] :identities [zero]
                  :dtypes [:float] :algebra [{}]}
             [array ...] [capture ...]
             (lambda [acc element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [step-result]))))]
       [program-result ...])

   Scan mode may be `:inclusive` or `:exclusive`; it is an explicit result-layout property. Map
   lambdas return tuples, so horizontal fusion remains functional rather than encoding
   secondary results as hidden stores. Caller-owned map outputs use an equation-fact
   `:result-storage` vector aligned with those functional results; logical SSA identity therefore
   stays separate from physical write identity and host return semantics. Captures are explicit
   operands with explicit scalar lambda parameters, so stable value IDs never leak into lexical
   expression binding. Lambda regions have one typed, ordered local-SSA spine. A local initializer
   may reference parameters, the map index, and earlier locals; results may reference all locals.
   This represents shared scalar work once without smuggling type inference into an emitter."
  (:require [clojure.set :as set]
            [pattern.nanopass.dialect :as dialect
             :refer [def-dialect]]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.scan :as scan-ir]))

(defn value-id?
  "Stable logical value IDs. Vector IDs are used by existing SSA-like program envelopes."
  [value]
  (or (symbol? value) (keyword? value) (vector? value)))

(defn equation-id?
  [value]
  (or (integer? value) (value-id? value)))

(defn extent?
  [value]
  (or (value-id? value) (and (integer? value) (not (neg? value)))))

(defn scalar-literal?
  [value]
  (or (nil? value)
      (number? value)
      (boolean? value)
      (string? value)
      (keyword? value)))

(defn extent-shape
  "Canonical rank-one shape for an extent value ID.

   AbstractValue dimensions already accept symbolic S-expressions. Compound stable IDs therefore
   use an explicit `(value id)` dimension instead of being confused with shape structure."
  [extent]
  [(cond
     (integer? extent) extent
     (symbol? extent) extent
     :else (list 'value extent))])

(defn scan-result-shape
  "The functional result shape of a scan operation.

   Inclusive scan produces one value per input element. Raster's public exclusive scan includes
   the initial identity at element zero, so its result has `extent + 1` elements. The logical
   traversal extent remains unchanged and is what schedules use for work decomposition."
  [{:keys [mode extent]}]
  [(if (= :exclusive mode)
     (if (integer? extent) (inc extent) (list 'clojure.core/inc extent))
     (first (extent-shape extent)))])

(defn map-attributes?
  [value]
  (and (map? value)
       (symbol? (:index value))
       (extent? (:extent value))
       (or (nil? (:attributes value)) (map? (:attributes value)))))

(defn scatter-attributes?
  [value]
  (and (map-attributes? value)
       (= :unique (:conflict value))))

(defn reduce-attributes?
  [value]
  (and (map-attributes? value)
       (vector? (:accumulators value))
       (seq (:accumulators value))
       (every? symbol? (:accumulators value))
       (= (count (:accumulators value)) (count (distinct (:accumulators value))))
       (vector? (:identities value))
       (vector? (:dtypes value))
       (vector? (:algebra value))
       (= (count (:accumulators value))
          (count (:identities value))
          (count (:dtypes value))
          (count (:algebra value)))
       (every? keyword? (:dtypes value))
       (every? map? (:algebra value))))

(defn scan-attributes?
  "Attributes for a certified scan dialect operation. The explicit mode is load-bearing because
   inclusive and exclusive scans have different observable result layouts."
  [value]
  (and (contains? #{:inclusive :exclusive} (:mode value))
       (reduce-attributes? (dissoc value :mode))
       (every? scan-ir/associative-scan? (:algebra value))))

(defn scalar-attributes?
  [value]
  (and (map? value)
       (vector? (:dtypes value))
       (seq (:dtypes value))
       (every? keyword? (:dtypes value))
       (or (nil? (:attributes value)) (map? (:attributes value)))))

(defn equation-facts?
  [value]
  (and (map? value)
       (set? (:effects value))
       (map? (:aliases value))
       (map? (:provenance value))
       (map? (:attributes value))))

(defn program-facts?
  [value]
  (and (map? value)
       (map? (:values value))
       (every? value-id? (keys (:values value)))
       (every? av/abstract-value? (vals (:values value)))
       (vector? (:inputs value))
       (map? (:equations value))
       (every? equation-id? (keys (:equations value)))
       (every? equation-facts? (vals (:equations value)))
       (set? (:effects value))
       (vector? (:diagnostics value))
       (map? (:provenance value))
       (map? (:attributes value))))

(def-dialect TypedSOAC
  (terminals [id value-id?]
             [eid equation-id?]
             [sym symbol?]
             [lit scalar-literal?]
             [sa scalar-attributes?]
             [ma map-attributes?]
             [xa scatter-attributes?]
             [ra reduce-attributes?]
             [ca scan-attributes?]
             [dt keyword?]
             [facts program-facts?])

  (Scalar [s :enforce]
          ?sym ?lit
          (if ?s:test ?s:then ?s:else)
          (do (?:+ s))
          (let* [(?:* ?sym:binding ?s:init)] ?s:body)
          (write ?s:destination-index ?s:predicate ?s:value)
          [(?:* s)]
          (.invk ?sym:impl (?:* s:args))
          (& (?sym:f (?:* s:args)) (? _ seq?)))

  (Local [d :enforce]
         (let-value ?sym:binding ?dt ?s:init))

  (Region [r :enforce]
          (region [(?:* d)] [(?:+ s)]))

  (Lambda [l :enforce]
          (lambda [(?:* ?sym:parameter)] ?r))

  (Operation [o :enforce]
             (scalar ?sa [(?:* ?id:capture)] ?l)
             (map ?ma [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (scatter ?xa [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (reduce ?ra [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (scan ?ca [(?:* ?id:array)] [(?:* ?id:capture)] ?l))

  (Equation [q :enforce]
            (= ?eid [(?:+ ?id:result)] ?o))

  (Program [p :enforce]
           (soac-program ?facts [(?:* q)] [(?:* ?id:output)]))

  (entry Program))

(defn program-form?
  [value]
  (and (seq? value) (= 'soac-program (first value)) (= 4 (count value))))

(defn facts [program] (second program))
(defn equations [program] (nth program 2))
(defn outputs [program] (nth program 3))

(defn operation-kind
  [equation]
  (first (nth equation 3)))

(defn operation-inputs
  "Ordered array and capture operands of an equation. Extent is a separate scalar operand."
  [equation]
  (let [operation (nth equation 3)]
    (if (= 'scalar (first operation))
      (vec (nth operation 2))
      (vec (concat (nth operation 2) (nth operation 3))))))

(defn operation-extent
  [equation]
  (let [operation (nth equation 3)]
    (when-not (= 'scalar (first operation))
      (:extent (second operation)))))

(defn operation-parts
  "Normalize one operation into semantic fields shared by validation and lowering."
  [equation]
  (let [operation (nth equation 3)
        kind (first operation)]
    (if (= 'scalar kind)
      (let [[_ attributes captures lambda] operation]
        {:kind kind :attributes attributes :arrays [] :captures captures :lambda lambda})
      (let [[_ attributes arrays captures lambda] operation]
        {:kind kind :attributes attributes :arrays arrays :captures captures :lambda lambda}))))

(defn write-form?
  "Whether a scatter-region result is one explicit conditional indexed write."
  [value]
  (and (seq? value) (= 'write (first value)) (= 4 (count value))))

(defn write-parts
  [value]
  (when (write-form? value)
    (let [[_ destination-index predicate written-value] value]
      {:destination-index destination-index :predicate predicate :value written-value})))

(defn local-value
  "Construct one explicitly typed scalar-region SSA definition."
  [id dtype init]
  (list 'let-value id dtype init))

(defn lambda-region
  "Construct the canonical scalar region used by every TypedSOAC lambda."
  [locals results]
  (list 'region (vec locals) (vec results)))

(defn lambda-form
  "Construct a canonical TypedSOAC lambda, with an optional ordered local-SSA spine."
  ([parameters results]
   (lambda-form parameters [] results))
  ([parameters locals results]
   (list 'lambda (vec parameters) (lambda-region locals results))))

(defn lambda-parts
  "Project a canonical lambda into parameters, typed locals, and ordered result expressions.

   Locals are returned as maps so compiler passes do not independently parse their S-expression
   spelling. The TypedSOAC form remains the sole serialized representation."
  [lambda]
  (let [[_ parameters [_ local-forms body-results]] lambda]
    {:parameters (vec parameters)
     :locals (mapv (fn [[_ id dtype init]]
                     {:id id :dtype dtype :init init})
                   local-forms)
     :body-results (vec body-results)}))

(defn emit-locals
  "Return local-value forms for normalized local maps."
  [locals]
  (mapv (fn [{:keys [id dtype init]}] (local-value id dtype init)) locals))

(defn result-storage
  "Ordered physical storage contracts for an equation's functional results.

   A map that writes caller-owned buffers still defines fresh functional values. Each entry maps
   one equation result to the physical destination used when that result is materialized. Absence
   means the equation owns fresh result storage. The vector is aligned with the equation results;
   it is deliberately not a destination-name registry reconstructed by a backend."
  [program-or-facts equation-id]
  (get-in (if (program-form? program-or-facts)
            (facts program-or-facts)
            program-or-facts)
          [:equations equation-id :attributes :result-storage]))

(defn physical-results
  "Resolve an equation's ordered logical results to their physical storage identities."
  [program-or-facts equation]
  (let [results (vec (nth equation 2))
        storage (result-storage program-or-facts (second equation))]
    (if (seq storage)
      (mapv :destination storage)
      results)))

(defn parameter-layout
  "Split a SOAC lambda's ordered parameters into semantic roles."
  [equation]
  (let [{:keys [kind attributes arrays captures lambda]} (operation-parts equation)
        parameters (:parameters (lambda-parts lambda))
        accumulator-count (if (contains? #{'reduce 'scan} kind)
                            (count (:accumulators attributes)) 0)
        array-count (count arrays)
        accumulator-end accumulator-count
        element-end (+ accumulator-end array-count)]
    {:accumulators (subvec parameters 0 accumulator-end)
     :elements (subvec parameters accumulator-end element-end)
     :capture-parameters (subvec parameters element-end)}))

(defn- distinct-vector?
  [value]
  (and (vector? value) (= (count value) (count (distinct value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :dialect :typed-soac))))

(defn- validate-equation!
  [equation]
  (let [[_ equation-id results operation] equation
        {:keys [kind attributes arrays captures lambda]} (operation-parts equation)
        {:keys [parameters locals body-results]} (lambda-parts lambda)
        result-count (count results)]
    (when-not (distinct-vector? results)
      (fail! :typed-soac-equation-results "SOAC equation results must be distinct"
             {:equation equation-id :results results}))
    (doseq [[field ids] [[:arrays arrays] [:captures captures]]]
      (when-not (distinct-vector? ids)
        (fail! :typed-soac-operands "SOAC operands must be ordered and distinct"
               {:equation equation-id :field field :ids ids})))
    (when (seq (set/intersection (set arrays) (set captures)))
      (fail! :typed-soac-operand-role "one value cannot be both an element input and a capture"
             {:equation equation-id :arrays arrays :captures captures}))
    (let [stable-array-captures (get-in attributes [:attributes :stable-array-captures] [])]
      (when-not (and (distinct-vector? stable-array-captures)
                     (set/subset? (set stable-array-captures) (set captures)))
        (fail! :typed-soac-stable-array-captures
               "stable array capture roles must be an ordered subset of captures"
               {:equation equation-id :stable-array-captures stable-array-captures
                :captures captures})))
    (when-not (distinct-vector? parameters)
      (fail! :typed-soac-lambda-parameters "SOAC lambda parameters must be distinct"
             {:equation equation-id :parameters parameters}))
    (when-not (every? symbol? parameters)
      (fail! :typed-soac-lambda-parameters "SOAC lambda parameters must be symbols"
             {:equation equation-id :parameters parameters}))
    (let [local-ids (mapv :id locals)
          lexical-ids (vec (concat parameters local-ids))]
      (when-not (= (count lexical-ids) (count (distinct lexical-ids)))
        (fail! :typed-soac-region-binders
               "scalar-region parameters and local SSA definitions must be distinct"
               {:equation equation-id :parameters parameters :locals local-ids}))
      (when (and (contains? #{'map 'scatter 'reduce 'scan} kind)
                 (some #{(:index attributes)} local-ids))
        (fail! :typed-soac-region-binders
               "a scalar-region local cannot shadow its operation index"
               {:equation equation-id :index (:index attributes) :locals local-ids})))
    (doseq [{:keys [dtype] :as local} locals]
      ;; Region locals are currently materialized as typed JVM scalar bindings before the common
      ;; CPU/GPU scalar emitters see them. Reuse the compiler's authoritative dtype facets here;
      ;; accepting :half or an unknown keyword would only fail later in target lowering.
      (when-not (and (dtype/known? dtype)
                     (= dtype (dtype/canon dtype))
                     (:scalar-tag (dtype/info dtype)))
        (fail! :typed-soac-local-dtype
               "scalar-region locals require a supported JVM scalar dtype"
               {:equation equation-id :local local :dtype dtype})))
    (when (and (seq locals) (not (contains? #{'map 'scatter} kind)))
      (fail! :typed-soac-region-operation
             "typed local SSA is currently admitted only in map/scatter scalar regions"
             {:equation equation-id :operation kind :locals locals}))
    (when-not (= result-count (count body-results))
      (fail! :typed-soac-result-arity "SOAC result and lambda arity differ"
             {:equation equation-id :results result-count :body-results (count body-results)}))
    (let [expected-parameter-count (+ (if (contains? #{'reduce 'scan} kind)
                                        (count (:accumulators attributes))
                                        0)
                                      (count arrays)
                                      (count captures))]
      (when-not (= expected-parameter-count (count parameters))
        (fail! :typed-soac-lambda-arity
               "lambda parameters must cover accumulators, elements and captures in order"
               {:equation equation-id
                :expected expected-parameter-count
                :actual (count parameters)
                :arrays arrays
                :captures captures})))
    (let [initial-bound (cond-> (set parameters)
                          (contains? #{'map 'scatter 'reduce 'scan} kind)
                          (conj (:index attributes)))
          final-bound
          (reduce (fn [bound {:keys [id init] :as local}]
                    (let [unbound (util/free-syms init bound)]
                      (when (seq unbound)
                        (fail! :typed-soac-unbound-local
                               "local SSA initializers may reference only parameters and earlier locals"
                               {:equation equation-id :local local :unbound unbound}))
                      (conj bound id)))
                  initial-bound locals)]
      (doseq [body body-results
              expression (if (= 'scatter kind)
                           (vals (write-parts body))
                           [body])
              :let [unbound (util/free-syms expression final-bound)]
              :when (seq unbound)]
        (fail! :typed-soac-unbound-scalar
               "scalar-region results may reference only parameters and local SSA values"
               {:equation equation-id :unbound unbound :body body :expression expression})))
    (case kind
      scalar
      (when-not (= result-count (count (:dtypes attributes)))
        (fail! :typed-soac-scalar-results
               "scalar result arity must equal its declared dtype arity"
               {:equation equation-id :results results :dtypes (:dtypes attributes)}))

      map
      (when (some write-form? body-results)
        (fail! :typed-soac-map-write
               "functional map results cannot contain indexed writes"
               {:equation equation-id :results body-results}))

      scatter
      (when-not (every? write-form? body-results)
        (fail! :typed-soac-scatter-write
               "scatter results must be explicit conditional indexed writes"
               {:equation equation-id :results body-results}))

      reduce
      (let [accumulators (:accumulators attributes)]
        (when-not (= accumulators (vec (take (count accumulators) parameters)))
          (fail! :typed-soac-reduce-accumulators
                 "reduce accumulator parameters must lead the lambda in declared order"
                 {:equation equation-id :accumulators accumulators :parameters parameters}))
        (when-not (= result-count (count accumulators))
          (fail! :typed-soac-reduce-results
                 "reduce result arity must equal accumulator arity"
                 {:equation equation-id :results results :accumulators accumulators})))

      scan
      (let [accumulators (:accumulators attributes)]
        (when-not (= accumulators (vec (take (count accumulators) parameters)))
          (fail! :typed-soac-scan-accumulators
                 "scan accumulator parameters must lead the lambda in declared order"
                 {:equation equation-id :accumulators accumulators :parameters parameters}))
        (when-not (= result-count (count accumulators))
          (fail! :typed-soac-scan-results
                 "scan result arity must equal accumulator arity"
                 {:equation equation-id :results results :accumulators accumulators}))
        (doseq [[accumulator identity dtype certificate result]
                (map vector accumulators (:identities attributes) (:dtypes attributes)
                     (:algebra attributes) body-results)]
          (let [derived (scan-ir/certify {:acc accumulator :init identity :lambda result} dtype)]
            (when-not (= certificate derived)
              (fail! :typed-soac-scan-certificate
                     "scan algebra certificate disagrees with its scalar region"
                     {:equation equation-id :declared certificate :derived derived})))))

      (fail! :typed-soac-operation "unknown SOAC operation"
             {:equation equation-id :operation kind}))
    equation))

(defn- validate-equation-types!
  [values equation]
  (let [[_ equation-id results] equation
        {:keys [kind attributes arrays]} (operation-parts equation)
        extent (:extent attributes)
        stable-array-captures (get-in attributes [:attributes :stable-array-captures] [])]
    (doseq [id arrays]
      (let [value (get values id)]
        ;; Unknown IDs receive the more precise boundary diagnostic below.
        (when (and value
                   (not (and (= :tensor (:kind value)) (= (extent-shape extent) (:shape value)))))
          (fail! :typed-soac-array-type
                 "SOAC element operands must be rank-one tensors over the declared extent"
                 {:equation equation-id :id id :value value :extent extent}))))
    (doseq [id stable-array-captures]
      (let [value (get values id)]
        (when (and value
                   (not (and (= :tensor (:kind value)) (seq (:shape value)))))
          (when-not (= :resident-scalar-buffer (get-in value [:representation :kind]))
            (fail! :typed-soac-stable-array-type
                   "stable captures require tensor storage or an explicit resident scalar buffer"
                   {:equation equation-id :id id :value value})))))
    (case kind
      scalar
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value)) (= [] (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-scalar-result-type
                   "scalar results must be scalar tensors with their declared dtype"
                   {:equation equation-id :id id :value value :dtype dtype}))))

      map
      (doseq [id results]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= (extent-shape extent) (:shape value)))))
            (fail! :typed-soac-map-result-type
                   "map results must be rank-one tensors over the declared extent"
                   {:equation equation-id :id id :value value :extent extent}))))

      scatter
      (doseq [id results]
        (let [value (get values id)]
          (when (and value (not= :tensor (:kind value)))
            (fail! :typed-soac-scatter-result-type
                   "scatter results require tensor storage contracts"
                   {:equation equation-id :id id :value value}))))

      reduce
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value)) (= [] (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-reduce-result-type
                   "reduce results must be scalar tensors with their declared accumulator dtype"
                   {:equation equation-id :id id :value value :dtype dtype}))))

      scan
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= (scan-result-shape attributes) (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-scan-result-type
                   "scan result shape must agree with its explicit inclusive/exclusive mode"
                   {:equation equation-id :id id :value value
                    :extent extent :mode (:mode attributes) :dtype dtype}))))

      nil)))

(defn- validate-result-storage!
  [program-facts equation]
  (let [[_ equation-id results] equation
        {:keys [kind]} (operation-parts equation)
        storage (result-storage program-facts equation-id)]
    (when storage
      (when-not (contains? #{'map 'scatter} kind)
        (fail! :typed-soac-result-storage-operation
               "physical result storage is valid only for map/scatter equations"
               {:equation equation-id :operation kind :storage storage}))
      (when-not (and (vector? storage)
                     (= (count results) (count storage))
                     (every? #(and (map? %)
                                   (value-id? (:destination %))
                                   (contains? #{:write :read-write} (:access %))
                                   (contains? #{:buffer :effect} (:host-return %)))
                             storage))
        (fail! :typed-soac-result-storage
               "result storage must align every map result with a typed destination contract"
               {:equation equation-id :results results :storage storage}))
      (let [destinations (mapv :destination storage)
            aliases (get-in program-facts [:equations equation-id :aliases])]
        (when-not (= (count destinations) (count (distinct destinations)))
          (fail! :typed-soac-result-storage-alias
                 "one pointwise equation may write each physical destination only once"
                 {:equation equation-id :destinations destinations}))
        (doseq [[result destination] (map vector results destinations)]
          (when-not (= destination (get aliases result))
            (fail! :typed-soac-result-storage-alias
                   "every stored functional result must explicitly alias its physical destination"
                   {:equation equation-id :result result :destination destination
                    :aliases aliases})))
        (when-not (contains? (get-in program-facts [:equations equation-id :effects])
                             :memory/write)
          (fail! :typed-soac-result-storage-effect
                 "physical result storage requires an explicit memory-write effect"
                 {:equation equation-id :storage storage}))
        (doseq [[result destination] (map vector results destinations)
                :let [logical (get-in program-facts [:values result])
                      physical (get-in program-facts [:values destination])]]
          (when-not physical
            (fail! :typed-soac-result-storage-value
                   "a physical result destination requires an AbstractValue"
                   {:equation equation-id :result result :destination destination}))
          (when (and logical physical
                     (not= (dissoc logical :shape) (dissoc physical :shape)))
            (fail! :typed-soac-result-storage-type
                   "logical result and physical destination storage types disagree"
                   {:equation equation-id :result result :destination destination
                    :logical logical :physical physical})))))))

(defn validate!
  "Validate syntax, typed value references, ordered SSA definitions and explicit effects.
   Returns the input program unchanged."
  [program]
  (when-not (program-form? program)
    (fail! :typed-soac-program "expected a soac-program form" {:program program}))
  (when-not (dialect/valid? TypedSOAC program)
    (fail! :typed-soac-syntax "program does not conform to the TypedSOAC pattern dialect"
           {:details (dialect/validate TypedSOAC program)}))
  (let [program-facts (facts program)
        values (:values program-facts)
        program-inputs (:inputs program-facts)
        program-outputs (outputs program)
        equations (equations program)
        equation-ids (mapv second equations)
        equation-fact-ids (set (keys (:equations program-facts)))]
    (doseq [[id value] values]
      (try
        (av/validate! value)
        (catch clojure.lang.ExceptionInfo exception
          (fail! :typed-soac-value "invalid AbstractValue in typed SOAC program"
                 {:id id :cause (ex-data exception)}))))
    (when-not (distinct-vector? program-inputs)
      (fail! :typed-soac-inputs "program inputs must be an ordered distinct vector"
             {:inputs program-inputs}))
    (when-not (distinct-vector? program-outputs)
      (fail! :typed-soac-outputs "program outputs must be an ordered distinct vector"
             {:outputs program-outputs}))
    (when-not (= (count equation-ids) (count (distinct equation-ids)))
      (fail! :typed-soac-equation-ids "equation IDs must be unique"
             {:equations equation-ids}))
    (when-not (= equation-fact-ids (set equation-ids))
      (fail! :typed-soac-equation-facts
             "equation fact keys must exactly match the expression spine"
             {:facts equation-fact-ids :equations (set equation-ids)}))
    (doseq [equation equations]
      (validate-equation! equation)
      (validate-equation-types! values equation)
      (validate-result-storage! program-facts equation))
    (let [definitions (mapcat #(nth % 2) equations)
          definition-set (set definitions)
          references (set (mapcat (fn [equation]
                                    (cond-> (operation-inputs equation)
                                      (value-id? (operation-extent equation))
                                      (conj (operation-extent equation))))
                                  equations))
          external (set/difference references definition-set)
          total-effects (reduce set/union #{}
                                (map :effects (vals (:equations program-facts))))]
      (when-not (= (count definitions) (count definition-set))
        (fail! :typed-soac-definitions "logical values may be defined by only one equation"
               {:definitions definitions}))
      (let [storage-destinations
            (set (mapcat (fn [equation]
                           (map :destination
                                (or (result-storage program-facts (second equation)) [])))
                         equations))]
        (doseq [id (set/union definition-set references storage-destinations
                              (set program-inputs) (set program-outputs))]
          (when-not (contains? values id)
            (fail! :typed-soac-unknown-value "SOAC program references an unknown value"
                   {:id id}))))
      (when-not (= external (set program-inputs))
        (fail! :typed-soac-input-boundary
               "program inputs must exactly name values used but not defined"
               {:declared (set program-inputs) :inferred external}))
      (loop [available (set program-inputs)
             [equation & remaining] equations]
        (when equation
          (let [equation-id (second equation)
                required (cond-> (set (operation-inputs equation))
                           (value-id? (operation-extent equation))
                           (conj (operation-extent equation)))
                missing (set/difference required available)]
            (when (seq missing)
              (fail! :typed-soac-use-before-definition
                     "equations may use only program inputs or earlier equation results"
                     {:equation equation-id :missing missing :available available}))
            (recur (into available (nth equation 2)) remaining))))
      (when-not (set/subset? (set program-outputs) definition-set)
        (fail! :typed-soac-output-boundary "program outputs must be equation results"
               {:outputs program-outputs :definitions definition-set}))
      (when-not (= total-effects (:effects program-facts))
        (fail! :typed-soac-effects "program effects must equal the union of equation effects"
               {:declared (:effects program-facts) :inferred total-effects}))))
  program)

(defn make
  "Construct and validate a typed SOAC program."
  [program-facts equation-forms program-outputs]
  (validate! (list 'soac-program program-facts (vec equation-forms) (vec program-outputs))))

(defn remap-values
  "Alpha-rename stable value IDs throughout a TypedSOAC program.

   Lambda parameters are lexical binders and are deliberately untouched. This is the bridge used
   when a typed algorithm enters ParallelProgram's SSA envelope: logical source binders become the
   envelope's authoritative value IDs without reparsing the scalar region or losing its facts."
  [program value-map]
  (let [program (validate! program)
        rename #(get value-map % %)
        rename-dimension
        (fn [dimension]
          (cond
            (contains? value-map dimension)
            (let [id (rename dimension)]
              (if (symbol? id) id (list 'value id)))

            (and (seq? dimension) (contains? #{'value 'unknown-dimension} (first dimension))
                 (= 2 (count dimension))
                 (contains? value-map (second dimension)))
            (list (first dimension) (rename (second dimension)))

            :else dimension))
        rename-value (fn [value] (update value :shape #(mapv rename-dimension %)))
        source-facts (facts program)
        values (reduce-kv (fn [result id value]
                            (let [id' (rename id)]
                              (when (contains? result id')
                                (fail! :typed-soac-remap-collision
                                       "value remapping collapses distinct TypedSOAC IDs"
                                       {:target id' :mapping value-map}))
                              (assoc result id' (rename-value value))))
                          {} (:values source-facts))
        equation-forms
        (mapv (fn [[equals equation-id results :as equation]]
                (let [{:keys [kind attributes arrays captures lambda]} (operation-parts equation)
                      attributes (cond-> attributes
                                   (seq (get-in attributes
                                                [:attributes :stable-array-captures]))
                                   (update-in [:attributes :stable-array-captures]
                                              #(mapv rename %)))
                      operation (if (= 'scalar kind)
                                  (list kind attributes (mapv rename captures) lambda)
                                  (list kind (update attributes :extent
                                                     #(if (value-id? %) (rename %) %))
                                        (mapv rename arrays) (mapv rename captures) lambda))]
                  (list equals equation-id (mapv rename results) operation)))
              (equations program))
        rename-aliases
        (fn [aliases]
          (into {} (map (fn [[left right]] [(rename left) (rename right)])) aliases))
        equation-facts
        (into {}
              (map (fn [[id equation-facts]]
                     [id (-> equation-facts
                             (update :aliases rename-aliases)
                             (cond-> (seq (get-in equation-facts
                                                  [:attributes :result-storage]))
                               (update-in [:attributes :result-storage]
                                          #(mapv (fn [storage]
                                                   (update storage :destination rename))
                                                 %))))]))
              (:equations source-facts))
        facts' (assoc source-facts
                      :values values
                      :inputs (mapv rename (:inputs source-facts))
                      :equations equation-facts)]
    (make facts' equation-forms (mapv rename (outputs program)))))

(defn default-equation-facts
  ([] (default-equation-facts {}))
  ([provenance]
   {:effects #{} :aliases {} :provenance provenance :attributes {}}))

(defn default-program-facts
  [{:keys [values inputs equations effects diagnostics provenance attributes]
    :or {values {} inputs [] equations {} effects #{} diagnostics [] provenance {} attributes {}}}]
  {:values values
   :inputs (vec inputs)
   :equations equations
   :effects effects
   :diagnostics (vec diagnostics)
   :provenance provenance
   :attributes attributes})
