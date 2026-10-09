(ns raster.compiler.passes.parallel.scalar-region-lower
  "Lower a closed KernelBody ScalarRegion to typed scalar SSA operations.

   The region boundary already owns value IDs, dtypes and tensor axis maps. This pass uses only
   the central intrinsic table and explicit KernelBody casts; target emitters receive no source
   expression to re-infer."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn from-typed-result-transform
  "Project a validated TypedSOAC result transform to the shared scheduled ScalarRegion.

   This is a mechanical alpha-boundary projection: dtypes, captures and the expression already
   belong to TypedSOAC. No source operator or type inference occurs here."
  [transform]
  (when transform
    (let [{:keys [parameters body-results]} (dialect/lambda-parts (:lambda transform))
          accumulator (first parameters)
          substitutions
          (into {}
                (concat (map (juxt :parameter :value) (:operands transform))
                        (map (juxt :parameter :value) (:scalars transform))))]
      (body/->ScalarRegion
       (vec (concat [accumulator]
                    (map :value (:operands transform))
                    (map :value (:scalars transform))))
       (util/subst-syms substitutions (first body-results))
       (mapv #(-> % (assoc :sym (:value %)) (dissoc :value :parameter))
             (:operands transform))
       (:result-dtype transform)))))

(defn completed-floating-conversion?
  "Whether a region is precisely one explicit floating conversion of its accumulator.

   This narrow mixed-storage subset cannot move arithmetic into the destination precision.
   General mixed-width result expressions need independently typed intermediate operations."
  [region]
  (let [expression (:expression region)]
    (and (= 1 (count (:parameters region)))
         (empty? (:operands region))
         (seq? expression) (= 2 (count expression))
         (descriptor/cast-op? (first expression))
         (= (first (:parameters region)) (second expression))
         (contains? #{:float :double} (:result-dtype region))
         (= (:result-dtype region)
            (dtype/dtype-for-scalar-tag
             (descriptor/cast-result-tag (first expression)))))))

(defn make-region
  "Convert the target-neutral result-transform descriptor into KernelBody region data."
  [transform]
  (when transform
    (body/->ScalarRegion
     (vec (concat [(:acc transform)]
                  (map :sym (:operands transform))
                  (map :sym (:scalars transform))))
     (:expr transform)
     (vec (:operands transform))
     (get transform :dtype :float))))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data
                                 :reason :scalar-region-kernel-body-declined
                                 :missing-rule rule))))

(defn declined?
  [exception]
  (= :scalar-region-kernel-body-declined (:reason (ex-data exception))))

(defn- cast-policy
  [source target]
  (let [source (dtype/canon source)
        target (dtype/canon target)]
    (cond
      (= source target) nil
      (and (contains? #{:byte :int :long} source) (dtype/fp-dtype? target))
      [:nearest-even (if (= :half target) :ieee :exact)]

      (and (dtype/fp-dtype? source) (dtype/fp-dtype? target)
           (< (dtype/bytes-of source) (dtype/bytes-of target)))
      [:exact :exact]

      (and (dtype/fp-dtype? source) (dtype/fp-dtype? target)
           (> (dtype/bytes-of source) (dtype/bytes-of target)))
      [:nearest-even :ieee]

      :else
      (decline! :result-transform-cast
                "portable result transform requires an explicit supported floating conversion"
                {:source source :target target}))))

(defn lower
  "Lower a completed result region through the shared retained-type scalar language.
   Axis maps remain owner-proved coordinates; scalar arithmetic and conversion widths are
   retained per expression rather than inherited from the final store dtype."
  [region {:keys [accumulator accumulator-dtype store-dtype parameters coordinate-lower predicate
                  id-prefix scalar-math]}]
  (let [accumulator-id (first (:parameters region))
        operands (into {} (map (juxt :sym identity)) (:operands region))
        scalar-ids (drop (inc (count operands)) (:parameters region))
        _ (doseq [[id operand] operands
                  :let [parameter (get parameters id)]]
            (when-not (and parameter
                           (or (and (= :input (:kind parameter))
                                    (contains? #{:operand :lhs :rhs :epilogue} (:role parameter)))
                               (and (= :inout (:kind parameter)) (= :result (:role parameter))))
                           (= (dtype/canon (get operand :dtype :float))
                              (dtype/canon (:dtype parameter))))
              (decline! :result-transform-operand
                        "result-transform operand lacks its typed KernelBody parameter"
                        {:operand operand :parameter parameter})))
        _ (doseq [id scalar-ids
                  :let [parameter (get parameters id)]]
            (when-not (and parameter (= :scalar (:kind parameter))
                           (contains? #{:parameter :epilogue} (:role parameter)))
              (decline! :result-transform-scalar
                        "result-transform scalar lacks its typed KernelBody parameter"
                        {:scalar id :parameter parameter})))
        expression (util/subst-syms {accumulator-id accumulator} (:expression region))
        scalar-types (cond-> (into {} (map (fn [id] [id (:dtype (get parameters id))])) scalar-ids)
                       (symbol? accumulator) (assoc accumulator accumulator-dtype))
        builder (scalar-expression/make-lowerer
                 {:arrays (set (keys operands))
                  :array-types (into {} (map (fn [[id operand]] [id (get operand :dtype :float)]))
                                     operands)
                  :scalar-types scalar-types
                  :scalar-math scalar-math
                  :source-region [expression accumulator (keys parameters)]
                  :require-source-types? true
                  :lower-index (fn [x _] x)
                  :owner-load-coordinates (fn [id]
                                            (let [coordinates (coordinate-lower (:map (get operands id)))]
                                              (if (vector? coordinates) coordinates [coordinates])))
                  :predicate predicate :conversion-policy cast-policy :decline! decline!
                  :id-prefix (str (when id-prefix (str id-prefix "-")) "result-transform")})
        lowered ((:lower builder) expression (dtype/canon (:result-dtype region)) {})
        stored ((:cast builder) lowered (dtype/canon store-dtype) expression)]
    {:operations (:operations stored)
     :result (:result stored)
     :result-dtype (:type stored)}))

(defn lower-region
  "Close a semantic ScalarRegion as validated, target-neutral scalar SSA for a store site."
  [region {:keys [accumulator accumulator-dtype store-dtype indices] :as options}]
  (let [{:keys [operations result result-dtype]} (lower region options)]
    (body/->ScalarSSARegion
     (:parameters region) (:operands region) (vec indices) (dtype/canon accumulator-dtype)
     operations result result-dtype)))
