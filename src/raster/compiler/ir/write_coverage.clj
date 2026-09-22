(ns raster.compiler.ir.write-coverage
  "Destination coverage from retained semantics, never from ABI write permission."
  (:require [clojure.walk :as walk]
            [raster.compiler.ir.extent-proof :as extent-proof]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.scalar-range :as ranges]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.index-expression :as index-expression]))

(defn dense-result-covers?
  "Whether a validated dense functional result covers exactly capacity elements.
   resolve-dimension supplies the caller's checked concrete shape environment."
  [algorithm result capacity resolve-dimension]
  (when-let [equation (some #(when (some #{result} (nth % 2)) %) (soac/equations algorithm))]
    (when-let [shape (soac/dense-functional-result-shape algorithm equation result)]
      (let [dimensions (mapv resolve-dimension shape)]
        (and (every? #(and (integer? %) (<= 0 %)) dimensions)
             (= capacity (reduce *' 1 dimensions)))))))

(defn- substitute-captures
  [equation expression]
  (let [{:keys [captures]} (soac/operation-parts equation)
        {:keys [capture-parameters]} (soac/parameter-layout equation)]
    (walk/postwalk-replace (zipmap capture-parameters captures) expression)))

(defn- monomial-expression
  [{:keys [const factors]}]
  (when const (apply list 'clojure.core/* const factors)))

(defn- proof-extent
  [extent-environment expression]
  (or (some-> (and extent-environment
                   (extent-proof/product-monomial extent-environment expression))
              monomial-expression)
      expression))

(defn- dense-effect-shape
  [equation result extent-environment]
  (let [{:keys [kind attributes arrays captures destinations lambda]}
        (soac/operation-parts equation)
        {:keys [destination-parameters]} (soac/parameter-layout equation)
        result-index (.indexOf ^java.util.List (nth equation 2) result)
        target (nth destination-parameters result-index nil)
        substitute #(substitute-captures equation %)
        outer (:index attributes)
        outer-extent (substitute (:extent attributes))
        algebra-outer-extent (proof-extent extent-environment outer-extent)]
    (when (and (= 'effect-map kind) (empty? arrays) target
               (= (count destinations) (count (nth equation 2))))
      (letfn [(visit [region dimensions inherited-locals]
                (let [locals (into inherited-locals
                                   (map #(update % :init substitute)) (:locals region))]
                  (mapcat
                   (fn [effect]
                     (let [{:keys [region predicate loop index lower upper-bound extent lambda
                                   carry destination conflict destination-index]}
                           (soac/effect-parts effect)]
                       (cond
                         ;; Guarded stores are not a complete domain proof.
                         (and region predicate) [nil]
                         region (visit region dimensions locals)
                         loop
                         (if (and (nil? carry) (= 0 lower) (= :exclusive upper-bound)
                                  (not-any? #{index} (map first dimensions)))
                           (visit (soac/lambda-parts lambda)
                                  (conj dimensions [index (substitute extent)]) locals)
                           [nil])
                         (and (= target destination) (= :unique conflict)
                              (contains? #{true 1} predicate))
                         [{:dimensions dimensions
                           :form (algebra/index-form (substitute destination-index)
                                                     outer algebra-outer-extent locals
                                                     (into {} (rest dimensions)))}]
                         :else [nil])))
                   (:body-results region))))]
        (let [stores (vec (visit (soac/lambda-parts lambda)
                                 [[outer outer-extent]] []))]
          (when (and (seq stores) (every? some? stores)
                     (apply = (map :dimensions stores))
                     (algebra/dense-translated-forms? (mapv :form stores)))
            (cond-> (mapv second (:dimensions (first stores)))
              (< 1 (count stores)) (conj (count stores)))))))))

(defn- dense-scatter-shape
  [equation result extent-environment]
  (let [{:keys [kind attributes lambda]} (soac/operation-parts equation)
        result-index (.indexOf ^java.util.List (nth equation 2) result)
        {:keys [locals body-results]} (soac/lambda-parts lambda)
        write (some-> (nth body-results result-index nil) soac/write-parts)
        substitute #(substitute-captures equation %)
        outer (:index attributes)
        extent (substitute (:extent attributes))
        algebra-extent (proof-extent extent-environment extent)
        locals (mapv #(update % :init substitute) locals)
        form (when write
               (algebra/index-form (substitute (:destination-index write))
                                   outer algebra-extent locals {}))]
    (when (and (= 'scatter kind) (= :unique (:conflict attributes))
               write (contains? #{true 1} (:predicate write))
               (algebra/dense? form))
      [extent])))

(defn operation-read-values
  "Ordered values whose contents an operation can observe.

   Array operands are conservatively reads.  For effect/scatter captures, the typed lexical
   parameter must actually occur in a local initializer or result expression; an unused caller-
   owned destination passed by the source API is not a read merely because it remains in the
   call boundary.  Other operation kinds retain the dialect's conservative input contract."
  [equation]
  (let [{:keys [kind arrays captures lambda]} (soac/operation-parts equation)]
    (if (and (contains? #{'effect-map 'scatter} kind) lambda)
      (let [{:keys [capture-parameters]} (soac/parameter-layout equation)
            {:keys [locals body-results effect-result]} (soac/lambda-parts lambda)
            body (concat (map :init locals) body-results (when effect-result [effect-result]))
            symbols (set (filter symbol? (tree-seq coll? seq body)))]
        (vec (concat arrays
                     (keep (fn [[value parameter]]
                             (when (contains? symbols parameter) value))
                           (map vector captures capture-parameters)))))
      (soac/operation-inputs equation))))

(defn symbolic-complete-write-shape
  "Return the exact logical write domain of one result, or nil when coverage is unproved.

   Functional SOACs use their declared dense result.  Ordered effect maps are admitted only when
   their unconditional unique stores form an exact mixed-radix image rooted at zero.  The result
   is a shape, not a physical-capacity claim; allocation and linker consumers must separately
   prove equal volume and non-aliasing."
  ([algorithm-or-facts equation result]
   (symbolic-complete-write-shape algorithm-or-facts equation result nil))
  ([algorithm-or-facts equation result extent-environment]
   (or (soac/dense-functional-result-shape algorithm-or-facts equation result)
       (let [facts (if (soac/program-form? algorithm-or-facts)
                     (soac/facts algorithm-or-facts) algorithm-or-facts)
             value (get-in facts [:values result])]
         (when (and (= {:kind :plain} (:representation value))
                    (nil? (:logical-layout value)))
           (or (dense-effect-shape equation result extent-environment)
               (dense-scatter-shape equation result extent-environment)))))))

(defn- expression-range [expression types intervals]
  (let [typed (index-expression/lower-typed
               expression (set (keys types)) types :long
               (fn [reason message data] (throw (ex-info message (assoc data :reason reason)))))]
    (ranges/typed-index-range typed types intervals)))

(defn- unique-scatter-covers?
  [equation result capacity scalar-values]
  (let [{:keys [attributes captures lambda]} (soac/operation-parts equation)
        {:keys [capture-parameters]} (soac/parameter-layout equation)
        {:keys [locals body-results]} (soac/lambda-parts lambda)
        result-index (.indexOf ^java.util.List (nth equation 2) result)
        write (some-> (nth body-results result-index nil) soac/write-parts)
        captured (merge scalar-values
                        (into {} (map vector capture-parameters (map scalar-values captures))))
        captured (into {} (filter (fn [[_ {:keys [value type]}]]
                                    (and (integer? value) (contains? #{:int :long} type)
                                         (ranges/literal value type)))) captured)
        types (update-vals captured :type)
        intervals (update-vals captured #(ranges/literal (:value %) (:type %)))
        constants (update-vals captured :value)
        concrete (fn [expression]
                   (walk/postwalk-replace constants expression))
        extent-range (expression-range (concrete (:extent attributes)) types intervals)
        extent (when (and extent-range (= (:lower extent-range) (:upper extent-range))
                          (pos? (:lower extent-range)))
                 (:lower extent-range))
        outer (:index attributes)
        locals (mapv #(update % :init concrete) locals)
        materialized-address
        (when write
          (reduce (fn [address {:keys [id init]}]
                    (let [init (walk/postwalk-replace constants init)]
                      (walk/postwalk-replace {id init} address)))
                  (concrete (:destination-index write)) locals))
        address-types (assoc types outer :long)
        address-intervals (cond-> intervals extent (assoc outer {:lower 0 :upper (dec extent)}))
        address-range (when (and extent materialized-address)
                        (expression-range materialized-address address-types address-intervals))
        form (when (and extent write)
               (algebra/index-form (concrete (:destination-index write)) outer extent locals {}))]
    (and (= :unique (:conflict attributes))
         write (contains? #{true 1} (:predicate write))
         (= extent capacity)
         address-range (<= 0 (:lower address-range)) (< (:upper address-range) capacity)
         (algebra/injective? form))))

(defn rectangular-effect-covers?
  "Coverage by one unconditional unique store over a positive rectangular loop domain.
   Retained typed arithmetic must prove exact-width in-bounds addresses. Mixed-radix algebra
   proves injection; domain cardinality equal to capacity then proves surjection. Unsupported
   widths, carries, branches and multiple-store unions conservatively decline. Both the
   functional unique-scatter form and the structured effect-map form carry this proof."
  [algorithm result capacity scalar-values]
  (try
    (when-let [equation (some #(when (some #{result} (nth % 2)) %) (soac/equations algorithm))]
      (let [{:keys [kind attributes arrays captures destinations lambda]}
            (soac/operation-parts equation)
            value (get-in (soac/facts algorithm) [:values result])]
        (when (and (= {:kind :plain} (:representation value)) (nil? (:logical-layout value)))
          (if (= 'scatter kind)
            (unique-scatter-covers? equation result capacity scalar-values)
            (when (and (= 'effect-map kind) (empty? arrays))
              (let [{parameters :parameters :as region} (soac/lambda-parts lambda)
                    result-index (.indexOf ^java.util.List (nth equation 2) result)
                    target (get parameters (+ (count captures) result-index))
                    captured (merge scalar-values
                                    (into {} (map vector parameters (map scalar-values captures))))
                    captured (into {} (filter (fn [[_ {:keys [value type]}]]
                                                (and (integer? value)
                                                     (contains? #{:int :long} type)
                                                     (ranges/literal value type))))
                                   captured)
                    types (update-vals captured :type)
                    intervals (update-vals captured #(ranges/literal (:value %) (:type %)))
                    constants (update-vals captured :value)
                    bound (fn [expression]
                            (when-let [{:keys [lower upper]}
                                       (expression-range expression types intervals)]
                              (when (and (= lower upper) (pos? lower)) lower)))
                    outer (:index attributes)
                    extent (bound (:extent attributes))]
                (when (and extent (not (contains? captured outer))
                           (not-any? #(contains? scalar-values %)
                                     (drop (count captures) parameters))
                           (= (count destinations) (count (nth equation 2))))
                  (letfn [(visit [region dimensions]
                            ;; Referenced address/bound locals are absent from the environment and
                            ;; decline. Value-only locals do not affect coverage.
                            (when (and (= 1 (count (:body-results region)))
                                       (not-any? (into (set (keys captured))
                                                       (map first dimensions))
                                                 (map :id (:locals region))))
                              (let [{:keys [region loop index lower extent lambda carry destination
                                            destination-index predicate conflict]}
                                    (soac/effect-parts (first (:body-results region)))]
                                (cond
                                  region (visit region dimensions)
                                  loop
                                  (when (and (nil? carry) (= 0 lower)
                                             (not (contains? captured index))
                                             (not-any? #{index} (map first dimensions)))
                                    (when-let [n (bound extent)]
                                      (visit (soac/lambda-parts lambda)
                                             (conj dimensions [index n]))))
                                  (and (= target destination) (= :unique conflict)
                                       (contains? #{true 1} predicate))
                                  (let [types (into types
                                                    (map (fn [[id _]] [id :long])) dimensions)
                                        intervals (into intervals
                                                        (map (fn [[id n]]
                                                               [id {:lower 0 :upper (dec n)}]))
                                                        dimensions)
                                        address-range (expression-range destination-index
                                                                        types intervals)
                                        ;; Cast erasure is safe ONLY after exact-width validation.
                                        form (when address-range
                                               (algebra/index-form
                                                (walk/postwalk-replace constants destination-index)
                                                outer (second (first dimensions)) []
                                                (into {} (rest dimensions))))]
                                    (and address-range (<= 0 (:lower address-range))
                                         (< (:upper address-range) capacity)
                                         (= capacity (reduce *' 1 (map second dimensions)))
                                         (algebra/injective? form)))))))]
                    (visit region [[outer extent]])))))))))
    (catch clojure.lang.ExceptionInfo _ false)
    (catch ArithmeticException _ false)))
