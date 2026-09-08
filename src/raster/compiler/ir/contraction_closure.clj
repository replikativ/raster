(ns raster.compiler.ir.contraction-closure
  "Lexical binding of canonical contraction facts into the typed value graph.

   The facts retain their axis and scalar binders. Only the ordered external values are
   renamed by SSA; no substitution into nested stage expressions is necessary. This is
   a checked closure over ContractionFacts, not another contraction representation."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.contraction-facts :as facts]))

(defn- fail! [rule data]
  (throw (ex-info "typed contraction closure is not proved"
                  (assoc data :reason :typed-soac-contraction :missing-rule rule))))

(defn attributes?
  [x]
  (and (map? x) (facts/facts? (:contraction x))
       (every? #(and (vector? %) (every? symbol? %)
                    (= (count %) (count (set %))))
               [(:array-parameters x) (:capture-parameters x)])))

(defn validate!
  "Validate lexical closure and the currently admitted static staged semantics.
   Numerical/layout extensions must retain their own contracts before admission."
  [{:keys [contraction array-parameters capture-parameters] :as attributes}]
  (when-not (attributes? attributes) (fail! :attributes {:attributes attributes}))
  (let [dependencies (facts/dependencies contraction)
        axes (vec (concat (:free-axes contraction) (:contract-axes contraction)))
        indices (mapv first axes)
        stage-list (:stages contraction)
        legality (stages/stages-legal? stage-list (:contract-axes contraction))]
    (when-not (and (seq (:free-axes contraction)) (> (count stage-list) 1)
                   (= (count indices) (count (set indices)))
                   (every? symbol? indices)
                   (every? #(and (integer? %) (pos? %)) (map second axes)))
      (fail! :static-domain {:axes axes :stages stage-list}))
    (when-not (:ok legality) (fail! :stage-legality legality))
    (when-not (and (contains? '#{+ clojure.core/+ raster.numeric/+} (:combine contraction))
                   (constant/zero-value? (:init contraction))
                   (nil? (:epilogue contraction))
                   (empty? (get-in contraction [:opts :decode]))
                   (not-any? :decode (:operands contraction)))
      (fail! :numerical-contract {:contraction contraction}))
    (when-not (and (= (:reads dependencies) (set array-parameters))
                   (= (:scalars dependencies) (set capture-parameters))
                   (empty? (set/intersection (set array-parameters) (set capture-parameters)))
                   (empty? (set/intersection (set indices)
                                            (set (concat array-parameters capture-parameters))))
                   (symbol? (:out contraction))
                   (not (contains? (set indices) (:out contraction)))
                   (not (contains? (:reads dependencies) (:out contraction))))
      (fail! :lexical-boundary {:dependencies dependencies :attributes attributes})))
  attributes)

(defn bindings
  "Map lexical array/capture parameters to ordered authoritative value IDs."
  [attributes arrays captures]
  (validate! attributes)
  (when-not (and (= (count arrays) (count (:array-parameters attributes)))
                 (= (count captures) (count (:capture-parameters attributes))))
    (fail! :binding-arity {:arrays arrays :captures captures :attributes attributes}))
  (zipmap (concat (:array-parameters attributes) (:capture-parameters attributes))
          (concat arrays captures)))

(defn validate-values!
  "Check independent storage types/capacities and scalar capture declarations.
   Logical result shape is the free-axis space, not a shared flattened operand extent."
  [attributes arrays captures values result]
  (let [bound (bindings attributes arrays captures)
        source (:contraction attributes)
        domain (into {} (concat (:free-axes source) (:contract-axes source)))
        core (map #(assoc % :dtype (:dtype source)
                           :map (facts/operand-axis-map source %)) (:operands source))
        operands (concat core (stages/lift-operands (:stages source)))
        output-type (dtype/canon (or (:out-dtype source) (:dtype (first (:stages source)))))]
    (doseq [{:keys [sym dtype] amap :map :as operand} operands]
      (let [pairs (vec (mapcat identity (:groups amap)))
            ids (mapv first pairs)
            value (get values (get bound sym))
            shape (:shape value)]
        (when-not (and (seq pairs) (= (count ids) (count (set ids)))
                       (every? #(= (second %) (get domain (first %))) pairs))
          (fail! :operand-map {:operand operand :domain domain}))
        (let [required (reduce *' 1 (map second pairs))]
          (when-not (and (dtype/known? dtype) (= :tensor (:kind value))
                       (= (dtype/canon dtype) (:dtype value))
                       (seq shape) (every? #(and (integer? %) (pos? %)) shape)
                       (>= (reduce *' 1 shape) required))
            (fail! :operand-storage {:operand operand :value value :required required})))))
    (doseq [id captures :let [value (get values id)]]
      (when-not (and (= :tensor (:kind value)) (= [] (:shape value))
                     (dtype/known? (:dtype value)))
        (fail! :scalar-capture {:id id :value value})))
    (when-not (and (= :tensor (:kind result)) (= output-type (:dtype result))
                   (= (mapv second (:free-axes source)) (:shape result)))
      (fail! :result-type {:result result :dtype output-type
                          :shape (mapv second (:free-axes source))})))
  attributes)
