(ns raster.compiler.passes.parallel.typed-soac-fusion
  "Pattern-declared fusion over the typed functional SOAC dialect.

   The pattern library recognizes the compact equation grammar. Raster code performs the legality
   checks and fact-table updates explicitly; failed candidates leave the immutable program intact.
   This first slice covers map→map, map→reduce and horizontal map fusion."
  (:require [clojure.set :as set]
            [pattern.nanopass.dialect :refer [from-dialect]]
            [pattern.r3.core :refer [rule success]]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.soac-dialect :as dialect]))

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
                    (reduce-equation-rule equation)
                    (scan-equation-rule equation)])]
    (let [lambda (:lambda (dialect/operation-parts equation))]
      (merge (dissoc matched :local-definitions)
             (dialect/lambda-parts lambda)))))

(defn- equation-references
  [equation]
  (let [{:keys [arrays captures attributes]} (equation-info equation)]
    (cond-> (vec (concat arrays captures))
      (:extent attributes) (conj (:extent attributes)))))

(defn- element-symbols
  [n]
  (mapv #(symbol (str "%element" %)) (range n)))

(defn- capture-symbols
  [n]
  (mapv #(symbol (str "%capture" %)) (range n)))

(defn- parameter-parts
  [info]
  (let [accumulator-count (if (contains? #{:reduce :scan} (:kind info))
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

(defn- remove-equation-fact
  [facts removed-id kept-id fused-kind]
  (let [removed (get-in facts [:equations removed-id])
        kept (get-in facts [:equations kept-id])
        constituent-facts
        (fn [id equation-facts]
          (or (get-in equation-facts [:attributes :fusion/constituents])
              {id (update equation-facts :attributes dissoc :fusion/constituents)}))
        constituents (merge (constituent-facts kept-id kept)
                            (constituent-facts removed-id removed))]
    (-> facts
        (update :equations dissoc removed-id)
        (assoc-in [:equations kept-id :attributes :fusion/constituents] constituents)
        (assoc-in [:equations kept-id :provenance]
                  (merge-provenance
                   (assoc (:provenance kept) :fused-equations [kept-id])
                   (assoc (:provenance removed) :fused-equations [removed-id])
                   fused-kind)))))

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

(defn- vertical-candidate
  [program]
  (let [equations (dialect/equations program)
        infos (mapv equation-info equations)
        uses (value-use-counts program)]
    (first
     (for [producer-index (range (count infos))
           consumer-index (range (inc producer-index) (count infos))
           :let [producer (nth infos producer-index)
                 consumer (nth infos consumer-index)
                 produced (first (:results producer))]
           :when (= :map (:kind producer))
           :when (= 1 (count (:results producer)))
           :when (= 1 (count (:body-results producer)))
           :when (empty? (:locals producer))
           :when (empty? (:locals consumer))
           :when (contains? #{:map :reduce} (:kind consumer))
           :when (= (:extent (:attributes producer)) (:extent (:attributes consumer)))
           :when (= 1 (get uses produced 0))
           :when (some #{produced} (:arrays consumer))
           :when (fusible-equation? program (:id producer))
           :when (fusible-equation? program (:id consumer))]
       {:producer-index producer-index :consumer-index consumer-index
        :producer producer :consumer consumer :produced produced}))))

(defn- fuse-vertical-once
  [program]
  (when-let [{:keys [producer-index consumer-index producer consumer produced]}
             (vertical-candidate program)]
    (let [producer (freshen-parameters producer "producer")
          consumer (freshen-parameters consumer "consumer")
          producer-parameters (parameter-parts producer)
          consumer-parameters (parameter-parts consumer)
          consumed-index (.indexOf ^java.util.List (:arrays consumer) produced)
          consumer-elements (:elements consumer-parameters)
          consumed-parameter (nth consumer-elements consumed-index)
          producer-expression (util/subst-syms
                               {(get-in producer [:attributes :index])
                                (get-in consumer [:attributes :index])}
                               (first (:body-results producer)))
          inlined-body (mapv #(util/subst-syms
                               {consumed-parameter producer-expression} %)
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
          canonical (canonical-operands raw-arrays raw-elements
                                        raw-captures raw-capture-parameters [] inlined-body)
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
          equations (-> (dialect/equations program)
                        (assoc consumer-index (emit-equation updated))
                        (->> (keep-indexed (fn [index equation]
                                             (when-not (= index producer-index) equation)))
                             vec))
          facts (remove-equation-fact (dialect/facts program) (:id producer) (:id consumer)
                                      (keyword (str "map-" (name (:kind consumer)))))]
      (rebuild-boundary program facts equations))))

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
                 left-results (set (:results left))
                 right-results (set (:results right))
                 left-uses (set (concat (:arrays left) (:captures left)))
                 right-uses (set (concat (:arrays right) (:captures right)
                                         [(:extent (:attributes right))]))]
           :when (= :map (:kind left) (:kind right))
           :when (empty? (:locals left))
           :when (empty? (:locals right))
           :when (= (:extent (:attributes left)) (:extent (:attributes right)))
           :when (empty? (set/intersection left-results right-uses))
           :when (empty? (set/intersection right-results left-uses))
           ;; Moving the right equation to the left's position must not move it before an
           ;; intervening producer. Program inputs have no producer index and are always ready.
           :when (every? (fn [value]
                           (let [index (get producer-index value)]
                             (or (nil? index) (< index left-index))))
                         right-uses)
           :when (fusible-equation? program (:id left))
           :when (fusible-equation? program (:id right))]
       {:left-index left-index :right-index right-index :left left :right right}))))

(defn- fuse-horizontal-once
  [program]
  (when-let [{:keys [left-index right-index left right]} (horizontal-candidate program)]
    (let [left (freshen-parameters left "left")
          right (freshen-parameters right "right")
          left-parameters (parameter-parts left)
          right-parameters (parameter-parts right)
          canonical (canonical-operands
                     (vec (concat (:arrays left) (:arrays right)))
                     (vec (concat (:elements left-parameters)
                                  (:elements right-parameters)))
                     (vec (concat (:captures left) (:captures right)))
                     (vec (concat (:capture-parameters left-parameters)
                                  (:capture-parameters right-parameters)))
                     []
                     (vec (concat (:body-results left) (:body-results right))))
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
          facts (remove-equation-fact (dialect/facts program) (:id right) (:id left)
                                      :horizontal-map)]
      (rebuild-boundary program facts equations))))

(defn fusion-fixpoint
  "Fuse legal typed SOAC equations to a fixpoint. Returns [program stats]."
  [program]
  (dialect/validate! program)
  (loop [program program
         vertical 0
         horizontal 0
         iterations 0]
    (if-let [fused (fuse-vertical-once program)]
      (recur fused (inc vertical) horizontal (inc iterations))
      (if-let [fused (fuse-horizontal-once program)]
        (recur fused vertical (inc horizontal) (inc iterations))
        [program {:vertical vertical
                  :horizontal horizontal
                  :iterations (inc iterations)}]))))
