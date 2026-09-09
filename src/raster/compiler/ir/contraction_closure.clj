(ns raster.compiler.ir.contraction-closure
  "Lexical binding of canonical contraction facts into the typed value graph.

   The facts retain their axis and scalar binders. Only the ordered external values are
   renamed by SSA; no substitution into nested stage expressions is necessary. This is
   a checked closure over ContractionFacts, not another contraction representation."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.contraction-facts :as facts]))

(defn- fail! [rule data]
  (throw (ex-info "typed contraction closure is not proved"
                  (assoc data :reason :typed-soac-contraction :missing-rule rule))))

(defn plain-storage?
  "The current closure has no view/representation lowering. Do not erase such facts."
  [value]
  (and (= {:kind :plain} (:representation value))
       (nil? (:logical-layout value))
       (nil? (:sharding value))))

(defn attributes?
  [x]
  (and (map? x) (facts/facts? (:contraction x))
       (every? #(and (vector? %) (every? symbol? %)
                    (= (count %) (count (set %))))
               [(:array-parameters x) (:capture-parameters x)])))

(defn result-expression
  "Resolve a result transform's declared operand maps without changing its scalar binders."
  [source]
  (let [epilogue (:epilogue source)]
    (descriptor/rewrite-aget-indices
     (:expr epilogue)
     (into {} (map (fn [{:keys [sym map]}] [sym (am/index-expr map)])) (:operands epilogue)))))

(defn validate-result-scalar-types!
  "Check explicit result-transform capture declarations against authoritative scalar types."
  [source scalar-types]
  (let [declarations (get-in source [:epilogue :scalars])
        ids (mapv :sym declarations)]
    (when-not (and (every? symbol? ids) (= (count ids) (count (set ids))))
      (fail! :result-transform-scalars {:declarations declarations}))
    (doseq [{:keys [sym dtype]} declarations]
      (when-not (and (dtype/known? dtype)
                     (= (dtype/canon dtype) (some-> (get scalar-types sym) dtype/canon)))
        (fail! :result-transform-scalar-type
               {:scalar sym :declared dtype :actual (get scalar-types sym)}))))
  source)

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
    (when-not (and (seq (:free-axes contraction)) (seq stage-list)
                   (= (count indices) (count (set indices)))
                   (every? symbol? indices)
                   (every? #(and (integer? %) (pos? %)) (map second axes)))
      (fail! :static-domain {:axes axes :stages stage-list}))
    (when-not (:ok legality) (fail! :stage-legality legality))
    (when-not (and (contains? '#{+ clojure.core/+ raster.numeric/+} (:combine contraction))
                   (constant/zero-value? (:init contraction))
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
      (fail! :lexical-boundary {:dependencies dependencies :attributes attributes}))
    (when-let [epilogue (:epilogue contraction)]
      (let [acc (:acc epilogue)
            external (set (concat array-parameters capture-parameters))
            visible (into (conj external acc) (map first (:free-axes contraction)))
            unbound (util/free-syms (result-expression contraction) visible)]
        (when-not (and (symbol? acc) (some? (:expr epilogue))
                       (not (contains? external acc))
                       (not (contains? (set indices) acc))
                       (not= (:out contraction) acc)
                       (or (nil? (:dtype epilogue)) (dtype/known? (:dtype epilogue)))
                       (empty? unbound))
          (fail! :result-transform-scope {:epilogue epilogue :unbound unbound :visible visible}))))
    (doseq [[index stage] (map-indexed vector stage-list)
            :when (:lift stage)]
      (let [visible (into (set (concat array-parameters capture-parameters ['inner]))
                          (map first (concat (:free-axes contraction)
                                             (take (inc index) (:contract-axes contraction)))))
            lift (stages/substitute-operand-indices
                  (:lift stage) (stages/stage-index-exprs stage-list))
            unbound (util/free-syms lift visible)]
        (when (seq unbound)
          (fail! :stage-lift-scope {:axis (:axis stage) :unbound unbound :visible visible})))))
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

(defn storage-requirements
  "Derive lexical storage requirements once from validated contraction maps.
   Frontend value construction and closure validation share this projection."
  [attributes]
  (validate! attributes)
  (let [source (:contraction attributes)
        domain (into {} (concat (:free-axes source) (:contract-axes source)))
        core (map #(assoc % :dtype (:dtype source)
                           :map (facts/operand-axis-map source %)) (:operands source))
        operands (concat (map #(vector % domain) core)
                         (mapcat (fn [index stage]
                                   (let [visible (into {} (concat (:free-axes source)
                                                                 (take (inc index) (:contract-axes source))))]
                                     (map #(vector % visible) (:operands stage))))
                                 (range) (:stages source))
                         (map #(vector % (into {} (:free-axes source)))
                              (get-in source [:epilogue :operands])))]
    (mapv (fn [[{:keys [sym dtype] amap :map :as operand} visible-domain]]
            (let [pairs (vec (mapcat identity (:groups amap)))
                  ids (mapv first pairs)]
              (when-not (and (seq pairs) (= (count ids) (count (set ids)))
                             (every? #(= (second %) (get visible-domain (first %))) pairs))
                (fail! :operand-map {:operand operand :domain visible-domain}))
              (when-not (dtype/known? dtype)
                (fail! :operand-storage-dtype {:operand operand}))
              {:parameter sym :dtype (dtype/canon dtype)
               :elements (reduce *' 1 (map second pairs))}))
          operands)))

(defn validate-values!
  "Check independent storage types/capacities and scalar capture declarations.
   Logical result shape is the free-axis space, not a shared flattened operand extent."
  [attributes arrays captures values result]
  (let [bound (bindings attributes arrays captures)
        source (:contraction attributes)
        output-type (dtype/canon (or (:out-dtype source) (:dtype (first (:stages source)))))]
    (validate-result-scalar-types!
     source (into {} (map (fn [id] [id (:dtype (get values (bound id)))]))
                  (:capture-parameters attributes)))
    (doseq [{:keys [parameter dtype elements] :as requirement} (storage-requirements attributes)
            :let [value (get values (get bound parameter))
                  shape (:shape value)]]
      (when-not (and (= :tensor (:kind value)) (plain-storage? value)
                     (= dtype (:dtype value))
                     (seq shape) (every? #(and (integer? %) (pos? %)) shape)
                     (>= (reduce *' 1 shape) elements))
        (fail! :operand-storage {:operand requirement :value value :required elements})))
    (doseq [id captures :let [value (get values id)]]
      (when-not (and (= :tensor (:kind value)) (= [] (:shape value)) (plain-storage? value)
                     (dtype/known? (:dtype value)))
        (fail! :scalar-capture {:id id :value value})))
    (when-not (and (= :tensor (:kind result)) (plain-storage? result) (= output-type (:dtype result))
                   (= (mapv second (:free-axes source)) (:shape result)))
      (fail! :result-type {:result result :dtype output-type
                          :shape (mapv second (:free-axes source))})))
  attributes)
