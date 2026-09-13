(ns raster.compiler.ir.write-coverage
  "Destination coverage from retained semantics, never from ABI write permission."
  (:require [clojure.walk :as walk]
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

(defn- expression-range [expression types intervals]
  (let [typed (index-expression/lower-typed
               expression (set (keys types)) types :long
               (fn [reason message data] (throw (ex-info message (assoc data :reason reason)))))]
    (ranges/typed-index-range typed types intervals)))

(defn rectangular-effect-covers?
  "Coverage by one unconditional unique store over a positive rectangular loop domain.
   Retained typed arithmetic must prove exact-width in-bounds addresses. Mixed-radix algebra
   proves injection; domain cardinality equal to capacity then proves surjection. Unsupported
   widths, address locals, carries, branches and multiple-store unions conservatively decline."
  [algorithm result capacity scalar-values]
  (try
    (when-let [equation (some #(when (some #{result} (nth % 2)) %) (soac/equations algorithm))]
      (let [{:keys [kind attributes arrays captures destinations lambda]} (soac/operation-parts equation)
            value (get-in (soac/facts algorithm) [:values result])]
        (when (and (= 'effect-map kind) (empty? arrays)
                   (= {:kind :plain} (:representation value)) (nil? (:logical-layout value)))
          (let [{parameters :parameters :as region} (soac/lambda-parts lambda)
                result-index (.indexOf ^java.util.List (nth equation 2) result)
                target (get parameters (+ (count captures) result-index))
                captured (merge scalar-values
                                (into {} (map vector parameters (map scalar-values captures))))
                captured (into {} (filter (fn [[_ {:keys [value type]}]]
                                            (and (integer? value) (contains? #{:int :long} type)
                                                 (ranges/literal value type)))) captured)
                types (update-vals captured :type)
                intervals (update-vals captured #(ranges/literal (:value %) (:type %)))
                constants (update-vals captured :value)
                bound (fn [expression]
                        (when-let [{:keys [lower upper]} (expression-range expression types intervals)]
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
                                   (not-any? (into (set (keys captured)) (map first dimensions))
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
                                  (visit (soac/lambda-parts lambda) (conj dimensions [index n]))))
                              (and (= target destination) (= :unique conflict)
                                   (contains? #{true 1} predicate))
                              (let [types (into types (map (fn [[id _]] [id :long])) dimensions)
                                    intervals (into intervals (map (fn [[id n]] [id {:lower 0 :upper (dec n)}])) dimensions)
                                    address-range (expression-range destination-index types intervals)
                                    ;; Cast erasure is safe ONLY after exact-width range validation.
                                    form (when address-range
                                           (algebra/index-form
                                             (walk/postwalk-replace constants destination-index)
                                             outer (second (first dimensions)) []
                                             (into {} (rest dimensions))))]
                                (and address-range (<= 0 (:lower address-range))
                                     (< (:upper address-range) capacity)
                                     (= capacity (reduce *' 1 (map second dimensions)))
                                     (algebra/injective? form)))))))]
                (visit region [[outer extent]])))))))
    (catch clojure.lang.ExceptionInfo _ false)
    (catch ArithmeticException _ false)))
