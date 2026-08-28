(ns raster.compiler.passes.parallel.soac-dialect-adapter
  "Guarded compatibility projection from the current record SOAC graph to TypedSOAC.

   This guarded bridge feeds both differential tests and the first production scalar-reduction
   vertical. It accepts only maps, scalar/full reductions and the allocation/alias scaffolding
   produced by current horizontal fusion. Every unsupported legacy shape declines loudly so the
   bridge cannot silently erase compiler facts."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
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
  (let [arrays (vec (sort-by pr-str (:inputs node)))
        captures (vec (sort-by pr-str (:scalars node)))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        {:keys [results expressions]} (horizontal-map-parts node aliases)
        body-results (->> (elementize expressions arrays parameters (:idx node))
                          (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))]
    (list '= (:id node) results
          (list 'map {:index (:idx node) :extent (:bound node)}
                arrays captures
                (list 'lambda (vec (concat parameters capture-parameters)) body-results)))))

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
          arrays (vec (sort-by pr-str (:inputs node)))
          captures (vec (sort-by pr-str (:scalars node)))
          element-parameters (element-symbols (count arrays))
          capture-parameters (capture-symbols (count captures))
          body-results (->> (elementize (:results (reduction/fold-region product))
                                        arrays element-parameters (:index product))
                            (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))
          attributes {:index (:index product)
                      :extent (:bound node)
                      :accumulators [(:accumulator component)]
                      :identities [(:neutral component)]
                      :dtypes [(:dtype component)]
                      :algebra [(:algebra product)]}]
      (list '= (:id node) (vec (filter some? (reduction/results product)))
            (list 'reduce attributes arrays captures
                  (list 'lambda
                        (vec (concat [(:accumulator component)]
                                     element-parameters capture-parameters))
                        body-results))))))

(defn- operation-equation
  [node aliases]
  (cond
    (legacy/soac-map? node) (map-equation node aliases)
    (legacy/soac-reduce? node) (reduce-equation node)
    :else
    (fail! :unsupported-legacy-soac
           "legacy adapter supports only map and scalar/full reduce nodes"
           {:node (:id node) :type (.getName (class node))})))

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
  [equation default-dtype array-types]
  (let [[_ _ results operation] equation
        [kind attributes arrays captures] operation
        extent (:extent attributes)
        result-dtypes (if (= 'reduce kind)
                        (:dtypes attributes)
                        (repeat (count results) default-dtype))]
    (merge
     (if (dialect/value-id? extent) {extent (tensor-value :long [])} {})
     (into {} (map (fn [id]
                     [id (tensor-value (value-dtype id default-dtype array-types)
                                       (dialect/extent-shape extent))])
                   arrays))
     (into {} (map (fn [id]
                     [id (tensor-value (value-dtype id default-dtype array-types) [])])
                   captures))
     (into {} (map (fn [id dtype]
                     [id (tensor-value (or dtype default-dtype :double)
                                       (if (= 'reduce kind) []
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
   - :values — authoritative additional AbstractValues"
  [nodes {:keys [outputs dtype array-types values]
          :or {dtype :double array-types {} values {}}}]
  (let [nodes (ordered-nodes nodes)
        physical-outputs (reduce set/union #{}
                                 (keep #(when-not (legacy/scalar-binding? %)
                                          (legacy/soac-outputs %))
                                       nodes))
        unsupported (remove #(or (legacy/soac-map? %)
                                 (legacy/soac-reduce? %)
                                 (generated-scaffolding? % physical-outputs))
                            nodes)]
    (when-let [node (first unsupported)]
      (fail! :unsupported-legacy-node
             "legacy graph contains a node outside the differential adapter subset"
             {:node (:id node) :type (.getName (class node))}))
    (let [aliases (logical-aliases nodes)
          operation-nodes (filterv #(or (legacy/soac-map? %) (legacy/soac-reduce? %)) nodes)
          equations (mapv #(operation-equation % aliases) operation-nodes)
          equation-infos (mapv (fn [equation]
                                 {:id (second equation)
                                  :results (nth equation 2)
                                  :inputs (dialect/operation-inputs equation)
                                  :extent (dialect/operation-extent equation)})
                               equations)
          definitions (set (mapcat :results equation-infos))
          references (set (mapcat #(cond-> (:inputs %)
                                     (dialect/value-id? (:extent %)) (conj (:extent %)))
                                  equation-infos))
          inputs (vec (sort-by pr-str (set/difference references definitions)))
          outputs (vec (or outputs (:results (peek equation-infos)) []))
          inferred-values (reduce (fn [contracts equation]
                                    (reduce-kv merge-value contracts
                                               (equation-values equation dtype array-types)))
                                  {}
                                  equations)
          values (reduce-kv merge-value inferred-values values)
          equation-facts (into {}
                               (map (fn [node]
                                      [(:id node)
                                       (dialect/default-equation-facts
                                        {:adapter :legacy-soac
                                         :legacy-soac-id (:id node)})]))
                               operation-nodes)
          facts (dialect/default-program-facts
                 {:values values
                  :inputs inputs
                  :equations equation-facts
                  :provenance {:adapter :legacy-soac}
                  :attributes {:compatibility-view true}})]
      (dialect/make facts equations outputs))))
