(ns raster.compiler.passes.parallel.structured-control-frontend
  "Functionalize one counted sequential host loop around a TypedSOAC body.

   This pass knows neither RK4 nor any numerical method. It recognizes a generic `dotimes`
   fixpoint, alpha-flattens its lexical body, replaces complete destination-writing parallel
   operations with logical SSA values, and accepts the state transition only when AbstractValue
   analysis proves a complete compatible array copy back into the carried state.

   The result is a TypedStructuredControl expression whose body enters the same TypedSOAC fusion
   and scheduling vertical as a loop-free program. Prefix/suffix source is retained only so a later
   mixed-program envelope can splice the typed fixpoint into the enclosing host computation."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as typed-fusion]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(def ^:private dotimes-heads '#{dotimes clojure.core/dotimes})

(defn- dotimes-form?
  [expression]
  (and (seq? expression)
       (contains? dotimes-heads (first expression))
       (vector? (second expression))
       (= 2 (count (second expression)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason
                                 :pass :structured-control-frontend))))

(defn- destination-write
  [expression]
  (cond
    (par/par-stencil-form? expression)
    (:out (par/extract-par-stencil-info expression))

    (par/par-map-form? expression)
    (let [{:keys [out offset]} (par/extract-par-map-info expression)]
      (when-not offset out))

    :else nil))

(defn- source-symbols
  [expression]
  (cond
    (symbol? expression) #{expression}
    (and (seq? expression) (= 'quote (first expression))) #{}
    (coll? expression) (reduce into #{} (map source-symbols expression))
    :else #{}))

(defn- fresh-allocation-bindings
  [pairs]
  (into #{}
        (keep (fn [[binder initializer]]
                (when (and (seq? initializer)
                           (descriptor/alloc-op?
                            (descriptor/semantic-op initializer)))
                  binder)))
        pairs))

(defn- arraycopy-destination
  [expression]
  (when (and (seq? expression)
             (contains? #{'System/arraycopy 'java.lang.System/arraycopy}
                        (descriptor/semantic-op expression))
             (= 5 (count (descriptor/call-args expression))))
    (nth (descriptor/call-args expression) 2)))

(defn- allocate-id
  [state prefix source-meta]
  (loop [ordinal (:ordinal state)]
    (let [id (with-meta (symbol (str prefix ordinal)) source-meta)]
      (if (contains? (:reserved state) id)
        (recur (inc ordinal))
        [(-> state
             (assoc :ordinal (inc ordinal))
             (update :reserved conj id))
         id]))))

(declare lower-value)

(defn- emit-value
  [state expression lexical source-meta]
  (let [lexical-expression (util/subst-syms lexical expression)
        destination (destination-write lexical-expression)
        copy-destination (arraycopy-destination lexical-expression)]
    (when (and destination
               (contains? (par/collect-aget-arrays lexical-expression) destination))
      (fail! :structured-loop-hidden-carry
             "a destination-writing scratch operation may not read its prior physical value"
             {:destination destination :expression lexical-expression}))
    (when (and destination (contains? (:written state) destination))
      (fail! :structured-loop-repeated-destination
             "one physical destination may be written only once in a functionalized iteration"
             {:destination destination :expression lexical-expression}))
    (when (and copy-destination (contains? (:written state) copy-destination))
      (fail! :structured-loop-mutated-carry
             "the copied loop carry may not also be a destination-written body buffer"
             {:destination copy-destination :expression lexical-expression}))
    (let [rewritten (util/subst-syms (:aliases state) lexical-expression)
          [state id] (allocate-id state "rstr_loop_value_" source-meta)
          state (-> state
                    (update :pairs conj [id rewritten])
                    (cond-> destination
                      (update :writes conj {:binding id :destination destination
                                            :expression rewritten})
                      destination (update :written conj destination)
                      destination (assoc-in [:aliases destination] id)))]
      [state id])))

(defn- lower-sequence
  [state expressions lexical]
  (reduce (fn [[state _] expression]
            (lower-value state expression lexical))
          [state nil]
          expressions))

(defn- lower-binding-form
  [state expression lexical]
  (let [[_ bindings & body] expression]
    (when-not (even? (count bindings))
      (fail! :structured-loop-bindings
             "nested loop-body binding form requires an even binding vector"
             {:expression expression}))
    (let [[state lexical]
          (reduce (fn [[state lexical] [binder initializer]]
                    (when-not (symbol? binder)
                      (fail! :structured-loop-binder
                             "functionalized loop bindings must be symbols"
                             {:binder binder :expression expression}))
                    (let [[state value] (lower-value state initializer lexical)]
                      [state (assoc lexical binder value)]))
                  [state lexical]
                  (partition 2 bindings))]
      (lower-sequence state body lexical))))

(defn- lower-value
  [state expression lexical]
  (cond
    (and (seq? expression) (contains? #{'let 'let*} (first expression)))
    (lower-binding-form state expression lexical)

    (and (seq? expression) (= 'do (first expression)))
    (lower-sequence state (rest expression) lexical)

    :else
    (emit-value state expression lexical (meta expression))))

(defn- flatten-loop-body
  [body reserved]
  (let [[state result]
        (lower-sequence {:pairs [] :writes [] :written #{} :aliases {}
                         :ordinal 0 :reserved reserved}
                        body {})]
    (assoc state :result result)))

(defn- values->array-types
  [values]
  (into {} (keep (fn [[id value]]
                   (when (seq (:shape value)) [id (:dtype value)]))) values))

(defn- values->scalar-types
  [values]
  (into {} (keep (fn [[id value]]
                   (when (= [] (:shape value)) [id (:dtype value)]))) values))

(defn- one-loop-binding
  [pairs]
  (let [matches (keep-indexed (fn [ordinal [binder initializer]]
                                (when (dotimes-form? initializer)
                                  {:ordinal ordinal :binder binder :form initializer}))
                              pairs)]
    (when (= 1 (count matches)) (first matches))))

(defn- add-iteration-value
  [program iteration]
  (let [facts (soac/facts program)
        iteration-value (av/tensor {:dtype :long :shape []})]
    (soac/make (update facts :values #(assoc % iteration iteration-value))
               (soac/equations program)
               (soac/outputs program))))

(defn- boundary-id
  [state prefix]
  (allocate-id state prefix nil))

(defn- order-body-boundary
  [program declared]
  (let [facts (soac/facts program)
        actual (set (:inputs facts))]
    (soac/make (assoc facts :inputs (filterv actual declared))
               (soac/equations program)
               (soac/outputs program))))

(defn form->structured-loop
  "Return a certified structured-loop decomposition for one flat outer binding form, or nil.

   A candidate must contain exactly one binding whose initializer is `dotimes`. The loop body may
   contain nested `let`/`let*`/`do`, pure TypedSOAC operations, and complete destination-writing
   stencil/map operations. Its last operation must be a proved full array copy into the carried
   state. Unsupported source shapes decline; contradictions after accepting a destination write
   fail loudly.

   `options` is the same retained type/AbstractValue environment accepted by the TypedSOAC and
   host AbstractValue frontends."
  [source {:keys [dtype values array-types scalar-types abstract-machine]
           :or {dtype :double values {} array-types {} scalar-types {}} :as options}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings & outer-body] source
          pairs (when (and (vector? bindings) (even? (count bindings)))
                  (vec (map vec (partition 2 bindings))))]
      (when-let [{:keys [ordinal binder form]} (and pairs (one-loop-binding pairs))]
        (let [[_ [iteration raw-trip-count] & loop-body] form
              trip-count (if (integer? raw-trip-count)
                           (max 0 raw-trip-count)
                           raw-trip-count)
              prefix-pairs (subvec pairs 0 ordinal)
              suffix-pairs (subvec pairs (inc ordinal))
              prefix-source (list 'let*
                                  (vec (mapcat identity prefix-pairs))
                                  (if (seq prefix-pairs) (ffirst (rseq prefix-pairs)) nil))
              reserved (source-symbols source)
              flattened (flatten-loop-body loop-body reserved)
              flattened-pairs (:pairs flattened)
              [copy-binding copy-expression] (peek flattened-pairs)
              body-pairs (when (seq flattened-pairs) (pop flattened-pairs))
              flattened-source (when (seq flattened-pairs)
                                 (list 'let* (vec (mapcat identity flattened-pairs))
                                       copy-binding))
              outer-analysis (host-av/analyze prefix-source options)
              body-analysis
              (when flattened-source
                (host-av/analyze
                 flattened-source
                 {:dtype dtype
                  :values (:values outer-analysis)
                  :array-types array-types
                  :scalar-types (assoc scalar-types iteration :long)}))
              copy-certificate (when body-analysis
                                 (host-av/full-array-copy body-analysis copy-expression))
              write-certificates
              (mapv #(host-av/full-array-write body-analysis (:binding %) (:expression %))
                    (:writes flattened))
              written-destinations (set (map :destination (:writes flattened)))
              fresh-allocations (fresh-allocation-bindings prefix-pairs)
              suffix-uses (reduce into #{}
                                  (map util/free-syms
                                       (concat (map second suffix-pairs) outer-body)))]
          (when (and (soac/extent? trip-count)
                     copy-certificate
                     (every? some? write-certificates))
            (when-not (every? fresh-allocations written-destinations)
              (fail! :structured-loop-scratch-ownership
                     "functionalized destination writes require fresh prefix allocations"
                     {:destinations written-destinations
                      :fresh-allocations fresh-allocations}))
            (when-let [escaping (seq (set/intersection written-destinations suffix-uses))]
              (fail! :structured-loop-scratch-escape
                     "functionalized scratch destinations may not escape the sequential loop"
                     {:destinations (set escaping)}))
            (let [carry-initial (:destination copy-certificate)
                  carry-result (:source copy-certificate)
                  typed-source (list 'let* (vec (mapcat identity body-pairs)) carry-result)
                  typed-values (:values body-analysis)
                  typed-program
                  (typed-frontend/form->program
                   typed-source
                   {:dtype dtype
                    :values (assoc typed-values iteration
                                   (av/tensor {:dtype :long :shape []}))
                    :array-types (merge (values->array-types typed-values) array-types)
                    :scalar-types (merge (values->scalar-types typed-values) scalar-types
                                         {iteration :long})})]
              (when typed-program
                (let [typed-program (add-iteration-value typed-program iteration)
                      [fused fusion-stats]
                      (typed-fusion/fusion-fixpoint typed-program abstract-machine)
                      body-inputs (:inputs (soac/facts fused))
                      invariant-outers (filterv #(not (contains? #{carry-initial iteration} %))
                                                body-inputs)
                      boundary-values (:values outer-analysis)]
                  (when (and (contains? boundary-values carry-initial)
                             (every? #(contains? boundary-values %) invariant-outers)
                             (or (integer? trip-count)
                                 (contains? boundary-values trip-count)))
                    (let [[state iteration-parameter] (boundary-id flattened
                                                                   "rstr_loop_iteration_")
                          [state carry-parameter] (boundary-id state "rstr_loop_carry_")
                          [state carry-output] (boundary-id state "rstr_loop_output_")
                          [state invariant-bindings]
                          (reduce (fn [[state bindings] outer]
                                    (let [[state parameter]
                                          (boundary-id state "rstr_loop_invariant_")]
                                      [state (conj bindings {:outer outer
                                                             :parameter parameter})]))
                                  [state []] invariant-outers)
                          renames (merge {iteration iteration-parameter
                                          carry-initial carry-parameter}
                                         (into {} (map (juxt :outer :parameter)
                                                       invariant-bindings)))
                          body (-> (soac/remap-values fused renames)
                                   (order-body-boundary
                                    (vec (concat [iteration-parameter]
                                                 (map :parameter invariant-bindings)
                                                 [carry-parameter]))))
                          outer-values (assoc boundary-values
                                              carry-output
                                              (get-in outer-analysis
                                                      [:values carry-initial]))
                          loop-facts
                          {:id (symbol (str (name binder) "-typed-fixpoint"))
                           :effects (:effects (soac/facts body))
                           :provenance {:source-dialect :closed-clojure
                                        :front-end :structured-control-frontend
                                        :source-binding binder}
                           :attributes {:association :sequential
                                        :trip-count-semantics :clamp-nonnegative}}
                          loop
                          (control/make
                           loop-facts [iteration-parameter trip-count]
                           invariant-bindings
                           [{:initial carry-initial :parameter carry-parameter
                             :result (get renames carry-result carry-result)
                             :output carry-output}]
                           body outer-values)]
                      {:loop loop
                       :loop-binding binder
                       :prefix-bindings prefix-pairs
                       :suffix-bindings suffix-pairs
                       :outer-body (vec outer-body)
                       :flattened-body-source flattened-source
                       :typed-body-source typed-source
                       :copy-certificate copy-certificate
                       :write-certificates write-certificates
                       :fusion-stats fusion-stats})))))))))))
