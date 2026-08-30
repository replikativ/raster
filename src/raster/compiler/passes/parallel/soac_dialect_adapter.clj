(ns raster.compiler.passes.parallel.soac-dialect-adapter
  "Guarded compatibility projection from the current record SOAC graph to TypedSOAC.

   This guarded bridge feeds both differential tests and the first production scalar-reduction
   vertical. It accepts only maps, scalar/full reductions and the allocation/alias scaffolding
   produced by current horizontal fusion. Every unsupported legacy shape declines loudly so the
   bridge cannot silently erase compiler facts."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]))

(def ^:private array-constructor-heads
  '#{double-array float-array int-array long-array
     clojure.core/double-array clojure.core/float-array
     clojure.core/int-array clojure.core/long-array})

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :adapter :legacy-soac->typed-soac))))

(defn- ordered-nodes
  [nodes]
  (->> (if (map? nodes) (vals nodes) nodes)
       (sort-by :id)
       vec))

(defn- generated-scaffolding?
  [node physical-outputs]
  (and (legacy/scalar-binding? node)
       (let [expression (:expr node)]
         (or (and (symbol? expression) (contains? physical-outputs expression))
             (and (contains? physical-outputs (:sym node))
                  (seq? expression)
                  (contains? array-constructor-heads (first expression)))))))

(defn- logical-aliases
  [nodes]
  (into {}
        (keep (fn [node]
                (when (and (legacy/scalar-binding? node) (symbol? (:expr node)))
                  [(:expr node) (:sym node)])))
        nodes))

(defn- element-symbols
  [n]
  (mapv #(symbol (str "%element" %)) (range n)))

(defn- capture-symbols
  [n]
  (mapv #(symbol (str "%capture" %)) (range n)))

(defn- elementize
  [expressions arrays parameters index]
  (mapv (fn [expression]
          (reduce (fn [body [array parameter]]
                    (fusion-support/substitute-aget body array index index parameter))
                  expression
                  (map vector arrays parameters)))
        expressions))

(defn- pointwise-input?
  [expressions array index]
  (let [reads (filter (fn [{:keys [sym]}]
                        (and (symbol? sym) (= (name array) (name sym))))
                      (mapcat descriptor/aget-reads expressions))]
    (and (seq reads)
         (every? #(= index (descriptor/unwrap-int-cast (:idx %))) reads))))

(defn- cast-result
  [cast expression]
  (if cast (list cast expression) expression))

(defn- horizontal-map-parts
  [node aliases]
  (let [body (:lambda node)]
    (if (and (seq? body) (= 'do (first body)))
      (let [statements (vec (rest body))
            primary-expression (peek statements)
            side-effects (pop statements)
            sides
            (mapv (fn [statement]
                    (when-not (descriptor/aset-call? statement)
                      (fail! :unsupported-legacy-map-body
                             "multi-result legacy map contains a non-aset side effect"
                             {:node (:id node) :statement statement}))
                    (let [arguments (vec (descriptor/call-args statement))
                          buffer (descriptor/aset-array-sym statement)]
                      (when-not (and (= 3 (count arguments)) (= (:idx node) (nth arguments 1)))
                        (fail! :unsupported-legacy-map-index
                               "horizontal map store is not pointwise in the map index"
                               {:node (:id node) :statement statement}))
                      {:result (get aliases buffer buffer)
                       :expression (nth arguments 2)}))
                  side-effects)]
        {:results (vec (cons (:sym node) (map :result sides)))
         :expressions (vec (cons (cast-result (:cast-fn node) primary-expression)
                                 (map :expression sides)))})
      {:results [(:sym node)]
       :expressions [(cast-result (:cast-fn node) body)]})))

(defn- map-equation
  [node aliases]
  (let [{:keys [results expressions]} (horizontal-map-parts node aliases)
        ;; Imperative map-void is represented as a functional map result plus an explicit
        ;; equation-level destination contract.  The destination is an unused stable capture here
        ;; so it participates in the typed boundary; scheduling, not the scalar lambda, owns the
        ;; write.  This keeps mutation out of the functional body.
        destination (when (:void? node) (:primary-out node))
        [pointwise stable] ((juxt filter remove)
                            #(pointwise-input? expressions % (:idx node))
                            (:inputs node))
        arrays (vec (sort-by pr-str pointwise))
        stable (cond-> (set stable) destination (conj destination))
        captures (vec (sort-by pr-str (distinct (concat stable (:scalars node)))))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        body-results (->> (elementize expressions arrays parameters (:idx node))
                          (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))]
    (list '= (:id node) results
          (list 'map {:index (:idx node) :extent (:bound node)
                      :attributes {:stable-array-captures (vec (sort-by pr-str stable))}}
                arrays captures
                (dialect/lambda-form (vec (concat parameters capture-parameters))
                                     body-results)))))

(defn- reduce-equation
  [node]
  (when (seq (:segment-axes node))
    (fail! :unsupported-segmented-reduction
           "legacy adapter covers scalar/full reductions only"
           {:node (:id node) :segment-axes (:segment-axes node)}))
  (let [product (:reduction node)]
    (when-not (reduction/scalar? product)
      (fail! :unsupported-product-reduction
             "legacy adapter covers one-component reductions only"
             {:node (:id node) :components (count (:components product))}))
    (let [component (first (:components product))
          expressions (:results (reduction/fold-region product))
          [pointwise stable] ((juxt filter remove)
                              #(pointwise-input? expressions % (:index product))
                              (:inputs node))
          arrays (vec (sort-by pr-str pointwise))
          captures (vec (sort-by pr-str (distinct (concat stable (:scalars node)))))
          element-parameters (element-symbols (count arrays))
          capture-parameters (capture-symbols (count captures))
          body-results (->> (elementize expressions arrays element-parameters (:index product))
                            (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))
          attributes {:index (:index product)
                      :extent (:bound node)
                      :attributes {:stable-array-captures (vec (sort-by pr-str stable))}
                      :accumulators [(:accumulator component)]
                      :identities [(:neutral component)]
                      :dtypes [(:dtype component)]
                      :algebra [(:algebra product)]}]
      (list '= (:id node) (vec (filter some? (reduction/results product)))
            (list 'reduce attributes arrays captures
                  (dialect/lambda-form
                   (vec (concat [(:accumulator component)]
                                element-parameters capture-parameters))
                   body-results))))))

(defn- scalar-dtype
  [node scalar-dtypes scalar-types]
  (let [result (:sym node)
        expression (:expr node)
        result-tag (types/sym-type-tag result)
        expression-tag (when (instance? clojure.lang.IObj expression)
                         (or (:raster.type/tag (meta expression)) (:tag (meta expression))))]
    (or (get scalar-types result)
        (when (symbol? expression) (get scalar-dtypes expression))
        (dtype/dtype-for-scalar-tag result-tag)
        (dtype/dtype-for-scalar-tag expression-tag))))

(defn- scalar-equation
  [node scalar-dtypes scalar-types]
  (let [expression (:expr node)
        captures (vec (sort-by pr-str (util/free-syms expression)))
        parameters (capture-symbols (count captures))
        dtype (scalar-dtype node scalar-dtypes scalar-types)]
    (when-not dtype
      (fail! :unsupported-scalar-binding
             "typed scalar equations require a statically known result dtype"
             {:node (:id node) :symbol (:sym node) :expression expression}))
    (list '= (:id node) [(:sym node)]
          (list 'scalar {:dtypes [dtype]} captures
                (dialect/lambda-form
                 parameters
                 [(util/subst-syms (zipmap captures parameters) expression)])))))

(defn- operation-equation
  [node aliases]
  (cond
    (legacy/soac-map? node) (map-equation node aliases)
    (legacy/soac-reduce? node) (reduce-equation node)
    :else
    (fail! :unsupported-legacy-soac
           "legacy adapter supports only map and scalar/full reduce nodes"
           {:node (:id node) :type (.getName (class node))})))

(defn- selected-scalar-symbols
  [nodes operation-equations outputs]
  (let [by-symbol (into {} (keep #(when (legacy/scalar-binding? %) [(:sym %) %])) nodes)
        roots (set (concat outputs
                           (mapcat (fn [equation]
                                     (cond-> (dialect/operation-inputs equation)
                                       (dialect/value-id? (dialect/operation-extent equation))
                                       (conj (dialect/operation-extent equation))))
                                   operation-equations)))]
    (loop [needed (set/intersection roots (set (keys by-symbol)))]
      (let [dependencies (set (mapcat #(util/free-syms (:expr (get by-symbol %))) needed))
            needed' (set/union needed (set/intersection dependencies (set (keys by-symbol))))]
        (if (= needed needed') needed (recur needed'))))))

(defn- tensor-value
  [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- value-dtype
  [id default-dtype array-types]
  (or (get array-types id)
      (when (symbol? id) (get array-types (symbol (name id))))
      default-dtype
      :double))

(defn- equation-values
  [equation default-dtype array-types known-values]
  (let [[_ _ results operation] equation
        {:keys [kind attributes arrays captures]} (dialect/operation-parts equation)
        extent (:extent attributes)
        stable-array-captures (set (get-in attributes [:attributes :stable-array-captures]))
        result-dtypes (case kind
                        scalar (:dtypes attributes)
                        reduce (:dtypes attributes)
                        (repeat (count results) default-dtype))]
    (merge
     (if (and extent (dialect/value-id? extent)) {extent (tensor-value :long [])} {})
     (into {} (map (fn [id]
                     [id (tensor-value (value-dtype id default-dtype array-types)
                                       (dialect/extent-shape extent))])
                   arrays))
     (if (= 'scalar kind)
       (into {} (keep (fn [id]
                        (when-let [value (or (get known-values id)
                                             (when-let [declared
                                                        (and (symbol? id)
                                                             (dtype/dtype-for-scalar-tag
                                                              (types/sym-type-tag id)))]
                                               (tensor-value declared [])))]
                          [id value])))
             captures)
       (into {} (map (fn [id]
                       [id (or (get known-values id)
                               (if (or (contains? stable-array-captures id)
                                       (contains? array-types id)
                                       (and (symbol? id)
                                            (contains? array-types (symbol (name id)))))
                                 (tensor-value (value-dtype id default-dtype array-types)
                                               [(list 'unknown-dimension id)])
                                 (tensor-value (value-dtype id default-dtype array-types) [])))])
                     captures)))
     (into {} (map (fn [id dtype]
                     [id (tensor-value (or dtype default-dtype :double)
                                       (if (contains? #{'scalar 'reduce} kind) []
                                           (dialect/extent-shape extent)))])
                   results result-dtypes)))))

(defn- merge-value
  [values id contract]
  (if-let [prior (get values id)]
    (if (= prior contract)
      values
      (fail! :legacy-value-conflict
             "legacy nodes infer incompatible AbstractValues for one logical ID"
             {:id id :first prior :second contract}))
    (assoc values id contract)))

(defn legacy-nodes->program
  "Project current map/reduce nodes into a verified TypedSOAC program.

   Options:
   - :outputs — logical program results (defaults to the last equation's results)
   - :dtype — fallback element dtype
   - :array-types — symbol-to-dtype overrides
   - :values — authoritative additional AbstractValues
   - :scalar-types — symbol-to-dtype overrides for scalar equations
   - :include-scalar-bindings? — emit needed pure scalar bindings as typed equations
   - :preserve-scalar-bindings? — compatibility option that allows but omits scalar bindings"
  [nodes {:keys [outputs dtype array-types scalar-types values include-scalar-bindings?
                 preserve-scalar-bindings?]
          :or {dtype :double array-types {} scalar-types {} values {}
               include-scalar-bindings? false preserve-scalar-bindings? false}}]
  (let [nodes (ordered-nodes nodes)
        physical-outputs (reduce set/union #{}
                                 (keep #(when-not (legacy/scalar-binding? %)
                                          (legacy/soac-outputs %))
                                       nodes))
        unsupported (remove #(or (legacy/soac-map? %)
                                 (legacy/soac-reduce? %)
                                 (generated-scaffolding? % physical-outputs)
                                 (and (or include-scalar-bindings? preserve-scalar-bindings?)
                                      (legacy/scalar-binding? %)))
                            nodes)]
    (when-let [node (first unsupported)]
      (fail! :unsupported-legacy-node
             "legacy graph contains a node outside the differential adapter subset"
             {:node (:id node) :type (.getName (class node))}))
    (let [aliases (logical-aliases nodes)
          operation-nodes (filterv #(or (legacy/soac-map? %) (legacy/soac-reduce? %)) nodes)
          operation-equations (mapv #(operation-equation % aliases) operation-nodes)
          requested-outputs (vec (or outputs
                                     (some-> operation-equations peek (nth 2)) []))
          selected-scalars (if include-scalar-bindings?
                             (selected-scalar-symbols nodes operation-equations requested-outputs)
                             #{})
          {:keys [equations equation-nodes]}
          (reduce
           (fn [{:keys [scalar-dtypes] :as state} node]
             (cond
               (and (legacy/scalar-binding? node) (contains? selected-scalars (:sym node)))
               (let [equation (scalar-equation node scalar-dtypes scalar-types)
                     dtype (first (:dtypes (second (nth equation 3))))]
                 (-> state
                     (update :equations conj equation)
                     (update :equation-nodes conj node)
                     (assoc-in [:scalar-dtypes (:sym node)] dtype)))

               (or (legacy/soac-map? node) (legacy/soac-reduce? node))
               (-> state
                   (update :equations conj (operation-equation node aliases))
                   (update :equation-nodes conj node))

               :else state))
           {:equations [] :equation-nodes [] :scalar-dtypes {}}
           nodes)
          equation-infos (mapv (fn [equation]
                                 {:id (second equation)
                                  :results (nth equation 2)
                                  :inputs (dialect/operation-inputs equation)
                                  :extent (dialect/operation-extent equation)})
                               equations)
          definitions (set (mapcat :results equation-infos))
          references (set (mapcat #(cond-> (:inputs %)
                                     (and (:extent %) (dialect/value-id? (:extent %)))
                                     (conj (:extent %)))
                                  equation-infos))
          inputs (vec (sort-by pr-str (set/difference references definitions)))
          outputs requested-outputs
          inferred-values (reduce (fn [contracts equation]
                                    (reduce-kv merge-value contracts
                                               (equation-values equation dtype array-types
                                                                contracts)))
                                  {}
                                  equations)
          values (reduce-kv merge-value inferred-values values)
          equation-facts
          (into {}
                (map (fn [node]
                       (let [destination (when (and (legacy/soac-map? node) (:void? node))
                                           (:primary-out node))]
                         [(:id node)
                          (cond-> (dialect/default-equation-facts
                                   {:adapter :legacy-soac :legacy-soac-id (:id node)})
                            destination
                            (assoc :effects #{:memory/write}
                                   :aliases {(:sym node) destination}
                                   :attributes {:destination destination}))])))
                equation-nodes)
          total-effects (reduce set/union #{} (map :effects (vals equation-facts)))
          facts (dialect/default-program-facts
                 {:values values
                  :inputs inputs
                  :equations equation-facts
                  :effects total-effects
                  :provenance {:adapter :legacy-soac}
                  :attributes {:compatibility-view true}})]
      (dialect/make facts equations outputs))))
