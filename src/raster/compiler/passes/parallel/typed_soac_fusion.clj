(ns raster.compiler.passes.parallel.typed-soac-fusion
  "Pattern-declared fusion over the typed functional SOAC dialect.

   The pattern library recognizes the compact equation grammar. Raster code performs the legality
   checks and fact-table updates explicitly; failed candidates leave the immutable program intact.
   This first slice covers map→map, map→reduce and horizontal map fusion."
  (:require [clojure.set :as set]
            [pattern.nanopass.dialect :refer [from-dialect]]
            [pattern.r3.core :refer [rule success]]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.fusion-placement :as placement]))

(def ^:private map-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (map ?attributes [??arrays] [??captures]
                               (lambda [??parameters]
                                       (region [??local-definitions] [??body-results]))))
                      (success {:kind :map
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private reduce-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (reduce ?attributes [??arrays] [??captures]
                                  (lambda [??parameters]
                                          (region [??local-definitions] [??body-results]))))
                      (success {:kind :reduce
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private segmented-reduce-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (segmented-reduce ?attributes [??arrays] [??captures]
                                            (lambda [??parameters]
                                                    (region [??local-definitions]
                                                            [??body-results]))))
                      (success {:kind :segmented-reduce
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private product-reduce-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (product-reduce ?attributes [??arrays] [??captures]
                                          ?element-lambda ?combine-lambda))
                      (success {:kind :product-reduce
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :element-lambda element-lambda
                                :combine-lambda combine-lambda}))))

(def ^:private segmented-fold-map-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (segmented-fold-map ?attributes [??arrays] [??captures]
                                              [??folds] ?map-lambda))
                      (success {:kind :segmented-fold-map
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :folds folds
                                :map-lambda map-lambda}))))

(def ^:private scatter-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (scatter ?attributes [??arrays] [??captures]
                                   (lambda [??parameters]
                                           (region [??local-definitions] [??body-results]))))
                      (success {:kind :scatter
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private stencil-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (stencil ?attributes [??arrays] [??captures]
                                   (lambda [??parameters]
                                           (region [??local-definitions] [??body-results]))))
                      (success {:kind :stencil
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private scalar-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (scalar ?attributes [??captures]
                                  (lambda [??parameters]
                                          (region [??local-definitions] [??body-results]))))
                      (success {:kind :scalar
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays []
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(def ^:private scan-equation-rule
  (from-dialect dialect/TypedSOAC
                (rule '(= ?equation-id [??results]
                          (scan ?attributes [??arrays] [??captures]
                                (lambda [??parameters]
                                        (region [??local-definitions] [??body-results]))))
                      (success {:kind :scan
                                :id equation-id
                                :results results
                                :attributes attributes
                                :arrays arrays
                                :captures captures
                                :local-definitions local-definitions
                                :body-results body-results}))))

(defn equation-info
  "Return a normalized equation description, using Pattern as the dialect matcher."
  [equation]
  (when-let [matched
             (some #(when (map? %) %)
                   [(scalar-equation-rule equation)
                    (map-equation-rule equation)
                    (scatter-equation-rule equation)
                    (stencil-equation-rule equation)
                    (reduce-equation-rule equation)
                    (segmented-reduce-equation-rule equation)
                    (product-reduce-equation-rule equation)
                    (segmented-fold-map-equation-rule equation)
                    (scan-equation-rule equation)])]
    (let [{:keys [lambda element-lambda map-lambda]} (dialect/operation-parts equation)]
      (merge (dissoc matched :local-definitions)
             (dialect/lambda-parts (or lambda element-lambda map-lambda))))))

(defn- equation-references
  [equation]
  (let [{:keys [arrays captures]} (equation-info equation)]
    (into (vec (concat arrays captures))
          (filter dialect/value-id? (dialect/operation-extents equation)))))

(defn- element-symbols
  [n]
  (mapv #(symbol (str "%element" %)) (range n)))

(defn- capture-symbols
  [n]
  (mapv #(symbol (str "%capture" %)) (range n)))

(defn- parameter-parts
  [info]
  (let [accumulator-count (if (contains? #{:reduce :segmented-reduce :scan} (:kind info))
                            (count (get-in info [:attributes :accumulators]))
                            0)
        array-count (count (:arrays info))
        parameters (vec (:parameters info))
        element-end (+ accumulator-count array-count)]
    {:accumulators (subvec parameters 0 accumulator-count)
     :elements (subvec parameters accumulator-count element-end)
     :capture-parameters (subvec parameters element-end)}))

(defn- canonical-operands
  "Sort/deduplicate operands and alpha-rename both element and capture parameters."
  [arrays element-parameters captures capture-parameters locals body-results]
  (let [canonical-arrays (vec (sort-by pr-str (distinct arrays)))
        canonical-elements (element-symbols (count canonical-arrays))
        element-for (zipmap canonical-arrays canonical-elements)
        canonical-captures (vec (sort-by pr-str (distinct captures)))
        canonical-capture-parameters (capture-symbols (count canonical-captures))
        capture-for (zipmap canonical-captures canonical-capture-parameters)
        substitutions
        (into {}
              (concat
               (map (fn [[array parameter]] [parameter (get element-for array)])
                    (map vector arrays element-parameters))
               (map (fn [[capture parameter]] [parameter (get capture-for capture)])
                    (map vector captures capture-parameters))))]
    {:arrays canonical-arrays
     :elements canonical-elements
     :captures canonical-captures
     :capture-parameters canonical-capture-parameters
     :locals (mapv #(update % :init (fn [init] (util/subst-syms substitutions init))) locals)
     :body-results (mapv #(util/subst-syms substitutions %) body-results)}))

(defn- stable-array-captures
  [info]
  (set (get-in info [:attributes :attributes :stable-array-captures])))

(defn- with-stable-array-captures
  [attributes captures stable]
  (assoc-in attributes [:attributes :stable-array-captures]
            (vec (filter stable captures))))

(defn- freshen-parameters
  "Give one equation's parameters and local SSA definitions private names before scopes combine."
  [info prefix]
  (let [{:keys [accumulators elements capture-parameters]} (parameter-parts info)
        fresh-elements (mapv #(symbol (str "%" prefix "-element" %))
                             (range (count elements)))
        fresh-captures (mapv #(symbol (str "%" prefix "-capture" %))
                             (range (count capture-parameters)))
        parameter-substitutions (merge (zipmap elements fresh-elements)
                                       (zipmap capture-parameters fresh-captures))
        {:keys [locals substitutions]}
        (reduce (fn [{:keys [locals substitutions]} {:keys [id dtype init]}]
                  (let [fresh-id (symbol (str "rstr_" prefix "_local_" (count locals)))]
                    {:locals (conj locals {:id fresh-id :dtype dtype
                                           :init (util/subst-syms substitutions init)})
                     :substitutions (assoc substitutions id fresh-id)}))
                {:locals [] :substitutions parameter-substitutions}
                (:locals info))]
    (assoc info
           :parameters (vec (concat accumulators fresh-elements fresh-captures))
           :locals locals
           :body-results (mapv #(util/subst-syms substitutions %) (:body-results info)))))

(defn- emit-equation
  [{:keys [kind id results attributes arrays captures parameters locals body-results]}]
  (list '= id (vec results)
        (list (symbol (name kind)) attributes (vec arrays) (vec captures)
              (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                   (vec body-results)))))

(defn- fusible-equation?
  [program equation-id]
  (let [facts (get-in (dialect/facts program) [:equations equation-id])]
    (and (empty? (:effects facts))
         ;; Alias-aware fusion can be added once legality is proved. Until then, equation results
         ;; are treated as fresh SSA values and any explicit alias contract declines fusion.
         (empty? (:aliases facts)))))

(declare remove-equation-fact)

(defn- horizontal-boundary
  "Return the proven write boundary of a horizontally fusible map, or nil.

   Pure SSA maps have an empty boundary. Materialized maps are also safe when
   every result has one distinct, dense unique-write destination. This is the
   alias-aware case produced by buffer fusion; it must remain explicit because
   combining two such maps combines their effects rather than erasing them."
  [program info]
  (let [equation-facts (get-in (dialect/facts program) [:equations (:id info)])
        effects (:effects equation-facts)
        aliases (:aliases equation-facts)
        storage (vec (get-in equation-facts [:attributes :result-storage]))
        destinations (mapv :destination storage)
        expected-aliases (zipmap (:results info) destinations)]
    (cond
      (and (empty? effects) (empty? aliases) (empty? storage))
      {:storage [] :destinations #{} :host-bindings {}}

      (and (= #{:memory/write} effects)
           (= (count (:results info)) (count (:body-results info)) (count storage))
           (every? #(and (= :write (:access %))
                         (contains? #{:buffer :effect} (:host-return %))
                         (symbol? (:destination %)))
                   storage)
           (= (count destinations) (count (distinct destinations)))
           (= expected-aliases aliases))
      (let [saved (get-in equation-facts [:attributes :host-bindings])
            singular (get-in equation-facts [:attributes :host-binding])
            buffer-results (map first
                                (filter (fn [[_ result-storage]]
                                          (= :buffer (:host-return result-storage)))
                                        (map vector (:results info) storage)))]
        {:storage storage
         :destinations (set destinations)
         :host-bindings
         (or saved
             (when (seq buffer-results)
               (into {}
                     (map-indexed (fn [index result]
                                    [result (if (zero? index) singular result)]))
                     buffer-results))
             {})})

      :else nil)))

(defn- merge-horizontal-facts
  [facts left right left-boundary right-boundary]
  (let [facts (remove-equation-fact facts (:id right) (:id left) :horizontal-map)
        left-facts (get-in facts [:equations (:id left)])
        storage (vec (concat (:storage left-boundary) (:storage right-boundary)))
        host-bindings (merge (:host-bindings left-boundary)
                             (:host-bindings right-boundary))
        aliases (merge (zipmap (:results left) (map :destination (:storage left-boundary)))
                       (zipmap (:results right) (map :destination (:storage right-boundary))))
        effects (if (seq storage) #{:memory/write} #{})]
    (-> facts
        (assoc-in [:equations (:id left) :effects] effects)
        (assoc-in [:equations (:id left) :aliases] aliases)
        (assoc-in [:equations (:id left) :attributes]
                  (cond-> (:attributes left-facts)
                    (seq storage) (assoc :result-storage storage
                                         :host-bindings host-bindings)
                    (empty? storage) (dissoc :result-storage :host-binding :host-bindings))))))

(defn- value-scalar-dtype
  "Return the canonical scalar dtype carried by one logical tensor value, or nil.

   Vertical region composition introduces an explicit local for the producer result so multiple
   consumer references do not duplicate its scalar expression. The local type must come from the
   existing value fact; fusion never infers or defaults it."
  [program value]
  (let [candidate (:dtype (get-in (dialect/facts program) [:values value]))]
    (when (and (keyword? candidate)
               (dtype/known? candidate)
               (= candidate (dtype/canon candidate))
               (:scalar-tag (dtype/info candidate)))
      candidate)))

(defn- value-use-counts
  [program]
  (frequencies
   (concat
    (mapcat equation-references (dialect/equations program))
    (dialect/outputs program))))

(defn- merge-provenance
  [left right fused-kind]
  (let [ids (vec (distinct (concat (:fused-equations left)
                                   (:fused-equations right))))]
    (assoc (merge left right)
           :fusion fused-kind
           :fused-equations ids)))

(defn- equation-constituents
  [equation-id equation-facts]
  (or (get-in equation-facts [:attributes :fusion/constituents])
      {equation-id (update equation-facts :attributes dissoc :fusion/constituents)}))

(defn- remove-equation-fact
  [facts removed-id kept-id fused-kind]
  (let [removed (get-in facts [:equations removed-id])
        kept (get-in facts [:equations kept-id])
        constituents (merge (equation-constituents kept-id kept)
                            (equation-constituents removed-id removed))]
    (-> facts
        (update :equations dissoc removed-id)
        (assoc-in [:equations kept-id :attributes :fusion/constituents] constituents)
        (assoc-in [:equations kept-id :provenance]
                  (merge-provenance
                   (assoc (:provenance kept) :fused-equations [kept-id])
                   (assoc (:provenance removed) :fused-equations [removed-id])
                   fused-kind)))))

(defn- record-recomputed-fusion
  "Record that `producer-id` was cloned into one consumer while its materialized equation remains."
  [facts producer-id consumer-id fused-kind placement-witness]
  (let [producer (get-in facts [:equations producer-id])
        consumer (get-in facts [:equations consumer-id])
        constituents (merge (equation-constituents consumer-id consumer)
                            (equation-constituents producer-id producer))]
    (-> facts
        (assoc-in [:equations consumer-id :attributes :fusion/constituents] constituents)
        (assoc-in [:equations consumer-id :attributes :fusion/placement] placement-witness)
        (assoc-in [:equations consumer-id :provenance]
                  (merge-provenance
                   (assoc (:provenance consumer) :fused-equations [consumer-id])
                   (assoc (:provenance producer) :fused-equations [producer-id])
                   fused-kind)))))

(defn- transfer-result-boundary
  "Keep `producer-id`'s fused provenance, but make the consumer's observable store its boundary."
  [facts producer-id consumer-id]
  (let [consumer (get-in facts [:equations consumer-id])
        facts (remove-equation-fact facts consumer-id producer-id
                                    :segmented-reduce-result-map)
        producer (get-in facts [:equations producer-id])
        boundary-attributes (select-keys (:attributes consumer)
                                         [:result-storage :host-binding])
        facts (assoc-in facts [:equations producer-id]
                        (assoc producer
                               :effects (:effects consumer)
                               :aliases (:aliases consumer)
                               :attributes (merge (:attributes producer)
                                                  boundary-attributes)))]
    (assoc facts :effects
           (reduce set/union #{} (map :effects (vals (:equations facts)))))))

(defn- rebuild-boundary
  [program facts equations]
  (let [definitions (set (mapcat #(nth % 2) equations))
        references (set (mapcat equation-references equations))
        inputs (vec (sort-by pr-str (set/difference references definitions)))
        storage-destinations
        (set (mapcat (fn [equation]
                       (map :destination
                            (or (dialect/result-storage facts (second equation)) [])))
                     equations))
        live-values (set/union definitions references storage-destinations
                               (set (dialect/outputs program)))
        facts (-> facts
                  (assoc :inputs inputs)
                  (update :values select-keys live-values))]
    (dialect/make facts equations (dialect/outputs program))))

(defn- single-write-boundary?
  [facts equation-id result destination]
  (let [equation-facts (get-in facts [:equations equation-id])
        storage (get-in equation-facts [:attributes :result-storage])]
    (and (= #{:memory/write} (:effects equation-facts))
         (= {result destination} (:aliases equation-facts))
         (= 1 (count storage))
         (= destination (:destination (first storage)))
         (= :write (:access (first storage))))))

(defn- result-map-transform
  "Translate one pointwise map region into a typed post-reduction scalar region.

   Pointwise arrays become full-segment operands. Stable captures retain only an axis map proven
   from their flat map index. Uniform captures become typed scalars. Any ambiguous array read
  declines the candidate rather than guessing an address."
  [program producer consumer consumed-destination]
  (let [segment-axes (get-in producer [:attributes :segment-axes])
        output-map (axis-map/of-axes segment-axes)
        output-extent (axis-map/n-elements output-map)
        flat-index (axis-map/index-expr output-map)
        map-index (get-in consumer [:attributes :index])
        producer-parameters (parameter-parts producer)
        consumer-parameters (parameter-parts consumer)
        accumulator (first (:accumulators producer-parameters))
        arrays (:arrays consumer)
        elements (:elements consumer-parameters)
        captures (:captures consumer)
        capture-parameters (:capture-parameters consumer-parameters)
        stable-values (stable-array-captures consumer)
        consumed-indices (keep-indexed #(when (= %2 consumed-destination) %1) arrays)
        consumed-index (first consumed-indices)
        expression (first (:body-results consumer))
        reads (descriptor/aget-reads expression)
        capture-bindings (mapv vector captures capture-parameters)
        stable-bindings (filterv #(contains? stable-values (first %)) capture-bindings)
        scalar-bindings (remove #(contains? stable-values (first %)) capture-bindings)
        indexed-operands
        (mapv (fn [[value parameter]]
                (let [indices (->> reads
                                   (keep #(when (= parameter (:sym %)) (:idx %)))
                                   distinct vec)]
                  (when (= 1 (count indices))
                    (when-let [operand-map
                               (axis-map/flat-index->map (first indices) map-index segment-axes)]
                      {:value value :dtype (value-scalar-dtype program value)
                       :map operand-map}))))
              stable-bindings)
        pointwise-operands
        (mapv (fn [index value]
                (when-not (= index consumed-index)
                  {:value value :dtype (value-scalar-dtype program value)
                   :map output-map}))
              (range) arrays)
        pointwise-operands (vec (remove nil? pointwise-operands))
        operands (vec (concat pointwise-operands indexed-operands))
        scalars (mapv (fn [[value _]]
                        {:value value :dtype (value-scalar-dtype program value)})
                      scalar-bindings)
        element-substitutions
        (into {}
              (map-indexed
               (fn [index parameter]
                 [parameter
                  (if (= index consumed-index)
                    accumulator
                    (list 'clojure.core/aget (nth arrays index) flat-index))])
               elements))
        capture-substitutions (into {} (map (fn [[value parameter]] [parameter value])
                                            capture-bindings))
        expression (->> expression
                        (util/subst-syms (merge element-substitutions
                                                capture-substitutions
                                                {map-index flat-index})))]
    (when (and (= output-extent (get-in consumer [:attributes :extent]))
               (= 1 (count consumed-indices))
               (every? some? indexed-operands)
               (every? :dtype operands)
               (every? :dtype scalars)
               (= (count (concat (map :value operands) (map :value scalars)))
                  (count (distinct (concat (map :value operands) (map :value scalars)))))
               (some? (value-scalar-dtype program (first (:results consumer)))))
      (dialect/make-result-transform
       {:accumulator accumulator
        :expression expression
        :operands operands
        :scalars scalars
        :result-dtype (value-scalar-dtype program (first (:results consumer)))}))))

(defn- segmented-reduce-result-map-candidate
  [program]
  (let [equations (dialect/equations program)
        infos (mapv equation-info equations)
        facts (dialect/facts program)
        uses (value-use-counts program)]
    (first
     (for [producer-index (range (dec (count infos)))
           :let [consumer-index (inc producer-index)
                 producer (nth infos producer-index)
                 consumer (nth infos consumer-index)]
           :when (= :segmented-reduce (:kind producer))
           :when (= :map (:kind consumer))
           :let [produced (first (:results producer))
                 consumed-destination
                 (get-in facts [:equations (:id producer)
                                :attributes :result-storage 0 :destination])
                 consumer-result (first (:results consumer))
                 consumer-destination
                 (get-in facts [:equations (:id consumer)
                                :attributes :result-storage 0 :destination])
                 transform (when (and consumed-destination consumer-destination)
                             (result-map-transform program producer consumer
                                                   consumed-destination))]
           :when (= 1 (count (:results producer)) (count (:body-results producer))
                    (count (:results consumer)) (count (:body-results consumer)))
           :when (= 1 (count (get-in producer [:attributes :accumulators])))
           :when (nil? (get-in producer [:attributes :result-transform]))
           :when (= (first (get-in producer [:attributes :dtypes]))
                    (value-scalar-dtype program consumer-result))
           :when (empty? (:locals consumer))
           :when (= 1 (get uses consumed-destination 0))
           :when (zero? (get uses produced 0))
           :when (not (contains? (set (dialect/outputs program)) produced))
           :when (not (contains? (set (dialect/outputs program)) consumed-destination))
           :when (= 1 (count (filter #(= consumed-destination %)
                                     (:arrays consumer))))
           :when (single-write-boundary? facts (:id producer) produced consumed-destination)
           :when (single-write-boundary? facts (:id consumer) consumer-result
                                         consumer-destination)
           :when (empty? (set/intersection
                          #{consumed-destination consumer-destination}
                          (set (map :value (:operands transform)))))
           :when transform]
       {:producer-index producer-index :consumer-index consumer-index
        :producer producer :consumer consumer :transform transform}))))

(defn- fuse-segmented-reduce-result-map-once
  [program]
  (when-let [{:keys [producer-index consumer-index producer consumer transform]}
             (segmented-reduce-result-map-candidate program)]
    (let [producer-parameters (parameter-parts producer)
          transform-values (vec (concat (map :value (:operands transform))
                                        (map :value (:scalars transform))))
          fresh-transform-parameters
          (mapv #(symbol (str "%fused-result-capture" %)) (range (count transform-values)))
          canonical (canonical-operands
                     (:arrays producer) (:elements producer-parameters)
                     (vec (concat (:captures producer) transform-values))
                     (vec (concat (:capture-parameters producer-parameters)
                                  fresh-transform-parameters))
                     (:locals producer) (:body-results producer))
          stable (set/union (stable-array-captures producer)
                            (set (map :value (:operands transform))))
          updated (assoc producer
                         :results (:results consumer)
                         :attributes (-> (:attributes producer)
                                         (with-stable-array-captures (:captures canonical) stable)
                                         (assoc :result-transform transform))
                         :arrays (:arrays canonical)
                         :captures (:captures canonical)
                         :parameters (vec (concat (:accumulators producer-parameters)
                                                  (:elements canonical)
                                                  (:capture-parameters canonical)))
                         :locals (:locals canonical)
                         :body-results (:body-results canonical))
          equations (-> (dialect/equations program)
                        (assoc producer-index (emit-equation updated))
                        (->> (keep-indexed (fn [index equation]
                                             (when-not (= index consumer-index) equation)))
                             vec))
          facts (-> (transfer-result-boundary (dialect/facts program)
                                              (:id producer) (:id consumer))
                    (update-in [:values (first (:results consumer))]
                               assoc
                               :shape (dialect/segmented-reduce-result-shape
                                       (:attributes updated))))]
      (rebuild-boundary program facts equations))))

(defn- producer-placement-witness
  [program infos uses producer produced abstract-machine]
  (let [consumer-ids (->> infos
                          (filter #(some #{produced} (equation-references
                                                      (emit-equation %))))
                          (mapv :id))
        witness (placement/placement-decision
                 {:abstract-machine abstract-machine
                  :dtype (value-scalar-dtype program produced)
                  :expressions (concat (map :init (:locals producer))
                                       (:body-results producer))
                  :consumer-count (get uses produced 0)})]
    (assoc witness
           :producer (:id producer)
           :value produced
           :consumers consumer-ids
           :externally-visible? (contains? (set (dialect/outputs program)) produced))))

(defn- vertical-candidates
  [program abstract-machine]
  (let [equations (dialect/equations program)
        infos (mapv equation-info equations)
        uses (value-use-counts program)]
    (vec
     (for [producer-index (range (count infos))
           consumer-index (range (inc producer-index) (count infos))
           :let [producer (nth infos producer-index)
                 consumer (nth infos consumer-index)
                 produced (first (:results producer))]
           :when (= :map (:kind producer))
           :when (= 1 (count (:results producer)))
           :when (= 1 (count (:body-results producer)))
           :when (contains? #{:map :reduce} (:kind consumer))
           ;; Local SSA is currently a map-region facility. A local-bearing producer/consumer can
           ;; therefore compose vertically into another map; the established local-free
           ;; map->reduce rule remains available until reduction regions admit typed locals.
           :when (or (= :map (:kind consumer))
                     (and (empty? (:locals producer)) (empty? (:locals consumer))))
           :when (or (= :reduce (:kind consumer))
                     (and (empty? (:locals producer)) (empty? (:locals consumer)))
                     (some? (value-scalar-dtype program produced)))
           :when (= (:extent (:attributes producer)) (:extent (:attributes consumer)))
           :when (pos? (get uses produced 0))
           :when (some #{produced} (:arrays consumer))
           :when (fusible-equation? program (:id producer))
           :when (fusible-equation? program (:id consumer))
           :let [placement-witness (producer-placement-witness
                                    program infos uses producer produced abstract-machine)]]
       {:producer-index producer-index :consumer-index consumer-index
        :producer producer :consumer consumer :produced produced
        :use-count (get uses produced 0)
        :placement placement-witness}))))

(defn- fuse-vertical-candidate
  [program {:keys [producer-index consumer-index producer consumer produced use-count]
            placement-witness :placement}]
    (let [producer (freshen-parameters producer "producer")
          consumer (freshen-parameters consumer "consumer")
          producer-parameters (parameter-parts producer)
          consumer-parameters (parameter-parts consumer)
          consumed-index (.indexOf ^java.util.List (:arrays consumer) produced)
          consumer-elements (:elements consumer-parameters)
          consumed-parameter (nth consumer-elements consumed-index)
          consumer-operation-index (get-in consumer [:attributes :index])
          producer-operation-index (get-in producer [:attributes :index])
          producer-locals (mapv #(update % :init
                                         (fn [init]
                                           (util/subst-syms
                                            {producer-operation-index
                                             consumer-operation-index} init)))
                                (:locals producer))
          producer-expression (util/subst-syms
                               {producer-operation-index consumer-operation-index}
                               (first (:body-results producer)))
          producer-local-ids (set (map :id producer-locals))
          local-region-composition? (or (seq producer-locals) (seq (:locals consumer)))
          producer-result-local (when (and (= :map (:kind consumer))
                                           local-region-composition?
                                           (not (contains? producer-local-ids
                                                           producer-expression)))
                                  {:id 'rstr_producer_result
                                   :dtype (value-scalar-dtype program produced)
                                   :init producer-expression})
          consumed-expression (if producer-result-local
                                (:id producer-result-local)
                                producer-expression)
          consumer-locals (mapv #(update % :init
                                         (fn [init]
                                           (util/subst-syms
                                            {consumed-parameter consumed-expression} init)))
                                (:locals consumer))
          inlined-body (mapv #(util/subst-syms
                               {consumed-parameter consumed-expression} %)
                             (:body-results consumer))
          raw-arrays (vec (concat (subvec (vec (:arrays consumer)) 0 consumed-index)
                                  (:arrays producer)
                                  (subvec (vec (:arrays consumer)) (inc consumed-index))))
          raw-elements (vec (concat (subvec consumer-elements 0 consumed-index)
                                    (:elements producer-parameters)
                                    (subvec consumer-elements (inc consumed-index))))
          raw-captures (vec (concat (:captures producer) (:captures consumer)))
          raw-capture-parameters
          (vec (concat (:capture-parameters producer-parameters)
                       (:capture-parameters consumer-parameters)))
          raw-locals (cond-> producer-locals
                       producer-result-local (conj producer-result-local)
                       true (into consumer-locals))
          canonical (canonical-operands raw-arrays raw-elements
                                        raw-captures raw-capture-parameters
                                        raw-locals inlined-body)
          stable (set/union (stable-array-captures producer)
                            (stable-array-captures consumer))
          updated (assoc consumer
                         :attributes (with-stable-array-captures
                                       (:attributes consumer) (:captures canonical) stable)
                         :arrays (:arrays canonical)
                         :captures (:captures canonical)
                         :parameters (vec (concat (:accumulators consumer-parameters)
                                                  (:elements canonical)
                                                  (:capture-parameters canonical)))
                         :locals (:locals canonical)
                         :body-results (:body-results canonical))
          retain-producer? (> use-count 1)
          fused-kind (keyword (str "map-" (name (:kind consumer))))
          equations (assoc (dialect/equations program) consumer-index (emit-equation updated))
          equations (if retain-producer?
                      equations
                      (->> equations
                           (keep-indexed (fn [index equation]
                                           (when-not (= index producer-index) equation)))
                           vec))
          facts (if retain-producer?
                  (record-recomputed-fusion (dialect/facts program)
                                            (:id producer) (:id consumer)
                                            fused-kind placement-witness)
                  (remove-equation-fact (dialect/facts program)
                                        (:id producer) (:id consumer) fused-kind))]
      (rebuild-boundary program facts equations)))

(defn- horizontal-candidate
  [program]
  (let [equations (dialect/equations program)
        infos (mapv equation-info equations)
        producer-index (into {}
                             (mapcat (fn [[index info]]
                                       (map (fn [result] [result index]) (:results info)))
                                     (map-indexed vector infos)))]
    (first
     (for [left-index (range (count infos))
           right-index (range (inc left-index) (count infos))
           :let [left (nth infos left-index)
                 right (nth infos right-index)
                 left-boundary (horizontal-boundary program left)
                 right-boundary (horizontal-boundary program right)
                 left-results (set (:results left))
                 right-results (set (:results right))
                 left-uses (set (concat (:arrays left) (:captures left)))
                 right-uses (set (concat (:arrays right) (:captures right)
                                         [(:extent (:attributes right))]))]
           :when (= :map (:kind left) (:kind right))
           :when left-boundary
           :when right-boundary
           :when (= (:extent (:attributes left)) (:extent (:attributes right)))
           :when (empty? (set/intersection left-results right-uses))
           :when (empty? (set/intersection right-results left-uses))
           :when (empty? (set/intersection (:destinations left-boundary)
                                           (:destinations right-boundary)))
           :when (empty? (set/intersection (:destinations left-boundary) right-uses))
           :when (empty? (set/intersection (:destinations right-boundary) left-uses))
           ;; Moving the right equation to the left's position must not move it before an
           ;; intervening producer. Program inputs have no producer index and are always ready.
           :when (every? (fn [value]
                           (let [index (get producer-index value)]
                             (or (nil? index) (< index left-index))))
                         right-uses)
           ]
       {:left-index left-index :right-index right-index :left left :right right
        :left-boundary left-boundary :right-boundary right-boundary}))))

(defn- fuse-horizontal-once
  [program]
  (when-let [{:keys [left-index right-index left right left-boundary right-boundary]}
             (horizontal-candidate program)]
    (let [left (freshen-parameters left "left")
          right (freshen-parameters right "right")
          left-parameters (parameter-parts left)
          right-parameters (parameter-parts right)
          right-operation-index (get-in right [:attributes :index])
          left-operation-index (get-in left [:attributes :index])
          right-locals (mapv #(update % :init
                                     (fn [init]
                                       (util/subst-syms
                                        {right-operation-index left-operation-index} init)))
                            (:locals right))
          right-results (mapv #(util/subst-syms
                                {right-operation-index left-operation-index} %)
                              (:body-results right))
          canonical (canonical-operands
                     (vec (concat (:arrays left) (:arrays right)))
                     (vec (concat (:elements left-parameters)
                                  (:elements right-parameters)))
                     (vec (concat (:captures left) (:captures right)))
                     (vec (concat (:capture-parameters left-parameters)
                                  (:capture-parameters right-parameters)))
                     (vec (concat (:locals left) right-locals))
                     (vec (concat (:body-results left) right-results)))
          stable (set/union (stable-array-captures left) (stable-array-captures right))
          updated (assoc left
                         :attributes (with-stable-array-captures
                                       (:attributes left) (:captures canonical) stable)
                         :results (vec (concat (:results left) (:results right)))
                         :arrays (:arrays canonical)
                         :captures (:captures canonical)
                         :parameters (vec (concat (:elements canonical)
                                                  (:capture-parameters canonical)))
                         :locals (:locals canonical)
                         :body-results (:body-results canonical))
          equations (-> (dialect/equations program)
                        (assoc left-index (emit-equation updated))
                        (->> (keep-indexed (fn [index equation]
                                             (when-not (= index right-index) equation)))
                             vec))
          facts (merge-horizontal-facts (dialect/facts program) left right
                                        left-boundary right-boundary)]
      (rebuild-boundary program facts equations))))

(defn- placement-key
  [{:keys [producer value decision]}]
  [producer value decision])

(defn- remember-placement
  [placements witness]
  (if (> (:consumer-count witness) 1)
    (assoc placements (placement-key witness) witness)
    placements))

(defn- attach-placement-facts
  [program placements]
  (if (seq placements)
    (let [ordered (->> (vals placements)
                       (sort-by (juxt (comp pr-str :producer)
                                      (comp pr-str :value)
                                      (comp name :decision)))
                       vec)
          facts (assoc-in (dialect/facts program)
                          [:attributes :fusion/placements] ordered)]
      [(dialect/make facts (dialect/equations program) (dialect/outputs program)) ordered])
    [program []]))

(defn fusion-fixpoint
  "Fuse legal typed SOAC equations to a fixpoint. Returns [program stats].

   With an Abstract Machine, legal multi-consumer map fusion is selected by the shared placement
   policy. Decisions are retained in program facts and stats; no device/vendor identity enters the
   typed dialect. The one-argument form preserves the existing single-consumer rewrites and keeps
   fan-out materialized when no target performance description can justify recomputation."
  ([program]
   (fusion-fixpoint program nil))
  ([program abstract-machine]
   (dialect/validate! program)
   (loop [program program
          vertical 0
          horizontal 0
          iterations 0
          placements {}]
     (let [candidates (vertical-candidates program abstract-machine)
           candidate (first (filter (comp :fuse? :placement) candidates))]
       (if candidate
         (recur (fuse-vertical-candidate program candidate)
                (inc vertical) horizontal (inc iterations)
                (remember-placement placements (:placement candidate)))
         (if-let [fused (fuse-segmented-reduce-result-map-once program)]
           (recur fused (inc vertical) horizontal (inc iterations) placements)
           (if-let [fused (fuse-horizontal-once program)]
             (recur fused vertical (inc horizontal) (inc iterations) placements)
             (let [placements
                   (reduce remember-placement placements
                           (map :placement
                                (filter (comp not :fuse? :placement) candidates)))
                   [program ordered-placements] (attach-placement-facts program placements)
                   stats {:vertical vertical
                          :horizontal horizontal
                          :iterations (inc iterations)}]
               [program (cond-> stats
                          (seq ordered-placements)
                          (assoc :placements ordered-placements))]))))))))
