(ns raster.compiler.ir.structured-control
  "Typed functional control around parallel algorithms.

   A sequential loop is not itself a SOAC.  It is a small fixpoint expression whose body is a
   closed TypedSOAC program.  Explicit outer-to-inner bindings keep lexical names out of compiler
   reasoning and make loop-carried state, invariants, zero-trip semantics, effects, and value
   compatibility independently certifiable before a schedule chooses host repetition or a
   persistent device kernel.

   Canonical form:

     (loop-program facts-with-outer-values
       [iteration-parameter trip-count]
       [{:outer invariant :parameter body-parameter} ...]
       [{:initial initial-value :parameter body-parameter
         :result body-result :output final-value} ...]
       typed-soac-body)

   `trip-count` is either a non-negative integer or an outer scalar value ID.  At zero trips each
   output denotes its corresponding initial value. The body may use any ordered subset of the
   iteration parameter, invariant parameters, and carried parameters; its ordered outputs are
   exactly the carried results."
  (:require [clojure.set :as set]
            [clojure.walk :as walk]
            [pattern.nanopass.dialect :as dialect :refer [def-dialect]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn loop-facts?
  [value]
  (and (map? value)
       (soac/equation-id? (:id value))
       (map? (:values value))
       (every? av/abstract-value? (vals (:values value)))
       (set? (:effects value))
       (map? (:provenance value))
       (map? (:attributes value))
       (= :sequential (get-in value [:attributes :association]))))

(defn loop-index?
  [value]
  (and (vector? value)
       (= 2 (count value))
       (soac/value-id? (first value))
       (soac/extent? (second value))))

(defn loop-invariant?
  [value]
  (and (map? value)
       (= #{:outer :parameter} (set (keys value)))
       (soac/value-id? (:outer value))
       (soac/value-id? (:parameter value))))

(defn loop-carry?
  [value]
  (and (map? value)
       (= #{:initial :parameter :result :output} (set (keys value)))
       (every? soac/value-id?
               ((juxt :initial :parameter :result :output) value))))

(def-dialect TypedStructuredControl
  (terminals [lf loop-facts?]
             [li loop-index?]
             [iv loop-invariant?]
             [lc loop-carry?]
             [sp soac/program-form?])

  (Program [p :enforce]
           (loop-program ?lf ?li [(?:* ?iv)] [(?:+ ?lc)] ?sp))

  (entry Program))

(defn loop-program?
  [value]
  (and (seq? value) (= 'loop-program (first value)) (= 6 (count value))))

(defn facts [program] (second program))
(defn loop-index [program] (nth program 2))
(defn invariants [program] (vec (nth program 3)))
(defn carried [program] (vec (nth program 4)))
(defn body [program] (nth program 5))

(defn outer-values
  "Retained enclosing AbstractValue environment used to certify every loop boundary."
  [program]
  (:values (facts program)))

(defn outer-operands
  "Ordered values read by one loop expression.  A symbolic trip count is an operand."
  [program]
  (let [[_ trip-count] (loop-index program)]
    (vec (concat (when (soac/value-id? trip-count) [trip-count])
                 (map :outer (invariants program))
                 (map :initial (carried program))))))

(defn outer-results
  [program]
  (mapv :output (carried program)))

(defn body-inputs
  "All declared loop-body binders in canonical scope order."
  [program]
  (vec (concat [(first (loop-index program))]
               (map :parameter (invariants program))
               (map :parameter (carried program)))))

(defn used-body-inputs
  "The minimal ordered input boundary used by the closed TypedSOAC body."
  [program]
  (vec (:inputs (soac/facts (body program)))))

(defn body-results
  [program]
  (mapv :result (carried program)))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :dialect :typed-structured-control))))

(defn- distinct-vector?
  [value]
  (and (vector? value) (= (count value) (count (distinct value)))))

(defn- integer-scalar?
  [value]
  (and (av/abstract-value? value)
       (= [] (:shape value))
       (contains? #{:int :long} (:dtype value))))

(defn- require-value!
  [values id role]
  (let [value (get values id ::missing)]
    (when (= ::missing value)
      (fail! :typed-loop-unknown-value "structured loop references an unknown value"
             {:id id :role role}))
    (av/validate! value)
    value))

(def ^:private remapped-value-facets
  [:shape :logical-layout :representation :placement :sharding :attributes])

(defn- remap-value
  "Rename scoped value IDs embedded in an AbstractValue without changing its record identity."
  [value renames]
  (reduce (fn [value facet]
            (update value facet #(walk/postwalk-replace renames %)))
          value remapped-value-facets))

(defn- require-compatible!
  [left-id left right-id right right->left role]
  (let [right (remap-value right right->left)]
    (when-not (= left right)
      (fail! :typed-loop-value-mismatch
             "structured loop boundary values require alpha-stable AbstractValue contracts"
             {:role role :left-id left-id :left left :right-id right-id :right right}))))

(defn validate!
  "Validate a typed structured loop and its retained enclosing AbstractValue environment."
  [program]
  (when-not (loop-program? program)
    (fail! :typed-loop-program "expected a loop-program form" {:program program}))
  (when-not (dialect/valid? TypedStructuredControl program)
    (fail! :typed-loop-syntax "program does not conform to the structured-control dialect"
           {:details (dialect/validate TypedStructuredControl program)}))
  (let [loop-facts (facts program)
        [iteration trip-count] (loop-index program)
        invariant-bindings (invariants program)
        carry-bindings (carried program)
        body-program (soac/validate! (body program))
        body-facts (soac/facts body-program)
        outer-values (:values loop-facts)
        inner-values (:values body-facts)
        expected-inputs (body-inputs program)
        expected-results (body-results program)
        inner-ids (vec (concat expected-inputs expected-results))
        outer-results (outer-results program)
        outer-operands (outer-operands program)
        inner->outer
        (into {}
              (concat (map (juxt :parameter :outer) invariant-bindings)
                      (map (juxt :parameter :initial) carry-bindings)
                      (map (juxt :result :output) carry-bindings)))]
    (when-not (distinct-vector? inner-ids)
      (fail! :typed-loop-inner-bindings
             "iteration, invariant, carry, and result IDs must be distinct inside the body"
             {:ids inner-ids}))
    (when-not (and (distinct-vector? outer-results)
                   (empty? (set/intersection (set outer-results)
                                             (set outer-operands))))
      (fail! :typed-loop-outer-bindings
             "loop results must be fresh and distinct from every enclosing operand"
             {:operands outer-operands :results outer-results}))
    (let [actual-inputs (:inputs body-facts)
          actual-set (set actual-inputs)
          expected-used (filterv actual-set expected-inputs)]
      (when-not (= expected-used actual-inputs)
        (fail! :typed-loop-body-inputs
               "loop body inputs must be an ordered subset of declared structured-loop binders"
               {:declared expected-inputs :expected-used expected-used
                :actual actual-inputs})))
    (when-not (= expected-results (soac/outputs body-program))
      (fail! :typed-loop-body-results
             "loop body outputs must exactly match the ordered carried results"
             {:expected expected-results :actual (soac/outputs body-program)}))
    (when-not (= (:effects loop-facts) (:effects body-facts))
      (fail! :typed-loop-effects "loop effects must equal its body effects"
             {:declared (:effects loop-facts) :body (:effects body-facts)}))
    (let [iteration-value (require-value! inner-values iteration :iteration)]
      (when-not (integer-scalar? iteration-value)
        (fail! :typed-loop-index-type "loop iteration parameter must be an integer scalar"
               {:id iteration :value iteration-value})))
    (doseq [{:keys [parameter]} invariant-bindings]
      (require-value! inner-values parameter :invariant-parameter))
    (doseq [{:keys [parameter result]} carry-bindings]
      (let [parameter-value (require-value! inner-values parameter :carry-parameter)
            result-value (require-value! inner-values result :carry-result)]
        (require-compatible! parameter parameter-value result result-value {} :body-carry)))
    (doseq [[_ value] outer-values]
      (av/validate! value))
    (when (soac/value-id? trip-count)
      (let [trip-value (require-value! outer-values trip-count :trip-count)]
        (when-not (integer-scalar? trip-value)
          (fail! :typed-loop-trip-count-type "symbolic trip count must be an integer scalar"
                 {:id trip-count :value trip-value}))))
    (doseq [{:keys [outer parameter]} invariant-bindings]
      (require-compatible!
       outer (require-value! outer-values outer :invariant)
       parameter (require-value! inner-values parameter :invariant-parameter)
       inner->outer :invariant))
    (doseq [{:keys [initial parameter result output]} carry-bindings]
      (let [initial-value (require-value! outer-values initial :carry-initial)
            parameter-value (require-value! inner-values parameter :carry-parameter)
            result-value (require-value! inner-values result :carry-result)
            output-value (require-value! outer-values output :carry-output)]
        (require-compatible! initial initial-value parameter parameter-value
                             inner->outer :carry-input)
        (require-compatible! output output-value result result-value
                             inner->outer :carry-output)))
    program))

(defn make
  "Construct and validate a typed structured loop."
  [loop-facts index invariant-bindings carry-bindings body-program outer-values]
  (validate!
   (list 'loop-program (assoc loop-facts :values outer-values)
         (vec index) (vec invariant-bindings) (vec carry-bindings) body-program)))
