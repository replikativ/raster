(ns raster.compiler.passes.parallel.scalar-region-lower
  "Lower a closed KernelBody ScalarRegion to typed scalar SSA operations.

   The region boundary already owns value IDs, dtypes and tensor axis maps. This pass uses only
   the central intrinsic table and explicit KernelBody casts; target emitters receive no source
   expression to re-infer."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.soac-dialect :as dialect]))

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
  "Lower `region` once per completed scalar reduction result.

   `parameters` maps region scalar/operand IDs to KernelParameters. `coordinate-lower` translates
   a declared operand axis-map to one or more scheduled storage coordinates. The returned `:result` is cast to
   `store-dtype`, so ScalarStore remains completely typed."
  [region {:keys [accumulator accumulator-dtype store-dtype parameters coordinate-lower predicate
                  id-prefix]}]
  (let [result-dtype (dtype/canon (:result-dtype region))
        accumulator-id (first (:parameters region))
        operand-by-id (into {} (map (juxt :sym identity)) (:operands region))
        scalar-ids (vec (drop (inc (count operand-by-id)) (:parameters region)))
        operations (atom [])
        counter (atom 0)
        fresh (fn [prefix]
                (symbol (str (when id-prefix (str id-prefix "-"))
                             prefix "-" (swap! counter inc))))
        emit! (fn [operation value value-dtype]
                (swap! operations conj operation)
                {:value value :dtype value-dtype})]
    (letfn [(cast [{:keys [value dtype] :as typed} target]
              (let [dtype (dtype/canon dtype)
                    target (dtype/canon target)]
                (if (= dtype target)
                  typed
                  (let [[rounding overflow] (cast-policy dtype target)
                        result (fresh "result-transform-cast")]
                    (emit! (body/->ScalarCompute
                            (body/value result target)
                            (body/cast-expression value target rounding overflow))
                           result target)))))

            (load-operand [id]
              (let [{:keys [map] :as operand} (get operand-by-id id)
                    operand-dtype (dtype/canon (get operand :dtype :float))
                    parameter (get parameters id)]
                (when-not (and operand parameter (= :input (:kind parameter))
                               (contains? #{:operand :lhs :rhs :epilogue} (:role parameter))
                               (= operand-dtype (dtype/canon (:dtype parameter))))
                  (decline! :result-transform-operand
                            "result-transform operand lacks its typed KernelBody parameter"
                            {:operand operand :parameter parameter}))
                (let [result (fresh "result-transform-load")
                      coordinates (coordinate-lower map)
                      coordinates (if (vector? coordinates) coordinates [coordinates])]
                  (cast
                   (emit! (body/->ScalarLoad
                           (body/value result operand-dtype) id coordinates
                           predicate
                           (when predicate (body/literal 0 operand-dtype))
                           :cached)
                          result operand-dtype)
                   result-dtype))))

            (lower-expression [expression]
              (cond
                (number? expression)
                {:value (body/literal expression result-dtype) :dtype result-dtype}

                (= accumulator-id expression)
                (cast {:value accumulator :dtype accumulator-dtype} result-dtype)

                (contains? (set scalar-ids) expression)
                (let [parameter (get parameters expression)]
                  (when-not (and parameter (= :scalar (:kind parameter))
                                 (contains? #{:parameter :epilogue} (:role parameter)))
                    (decline! :result-transform-scalar
                              "result-transform scalar lacks its typed KernelBody parameter"
                              {:scalar expression :parameter parameter}))
                  (cast {:value expression :dtype (:dtype parameter)} result-dtype))

                (descriptor/aget-call? expression)
                (let [operand (descriptor/aget-array-sym expression)]
                  (when-not (contains? operand-by-id operand)
                    (decline! :result-transform-load
                              "result-transform reads an undeclared tensor operand"
                              {:expression expression :operand operand}))
                  (load-operand operand))

                (and (seq? expression) (descriptor/cast-op? (first expression))
                     (= 2 (count expression)))
                (let [target (dtype/dtype-for-scalar-tag
                              (descriptor/cast-result-tag (first expression)))]
                  (when-not (= target result-dtype)
                    (decline! :result-transform-explicit-cast
                              "portable result transform requires casts to its declared result dtype"
                              {:expression expression :target target
                               :result-dtype result-dtype}))
                  (cast (lower-expression (second expression)) target))

                (seq? expression)
                (let [semantic-operation (descriptor/semantic-op expression)
                      operator (intrinsics/canonical semantic-operation)
                      intrinsic (intrinsics/descriptor operator)
                      arguments (vec (descriptor/call-args expression))]
                  (when-not (and intrinsic
                                 (= (:arity intrinsic) (count arguments))
                                 (not= :cmp (:kind intrinsic))
                                 (intrinsics/accepts-scalar-dtype? operator result-dtype))
                    (decline! :result-transform-expression
                              "result-transform expression has no typed portable scalar lowering"
                              {:expression expression :operator operator
                               :result-dtype result-dtype}))
                  (let [inputs (mapv (comp :value lower-expression) arguments)
                        overflow (when (and (contains? #{:byte :int :long} result-dtype)
                                            (contains? #{:+ :- :*} operator))
                                   (or (intrinsics/source-overflow-policy semantic-operation)
                                       :trap))
                        result (fresh "result-transform-value")]
                    (emit! (body/->ScalarCompute
                            (body/value result result-dtype)
                            (body/scalar-expression
                             operator result-dtype inputs
                             (cond-> {} overflow (assoc :overflow overflow))))
                           result result-dtype)))

                :else
                (decline! :result-transform-expression
                          "result-transform expression references an unbound or unsupported value"
                          {:expression expression})))]
      (let [typed-result (lower-expression (:expression region))
            stored-result (cast typed-result store-dtype)]
        {:operations @operations
         :result (:value stored-result)
         :result-dtype (:dtype stored-result)}))))

(defn lower-region
  "Close a semantic ScalarRegion as validated, target-neutral scalar SSA for a store site."
  [region {:keys [accumulator accumulator-dtype store-dtype indices] :as options}]
  (let [{:keys [operations result result-dtype]} (lower region options)]
    (body/->ScalarSSARegion
     (:parameters region) (:operands region) (vec indices) (dtype/canon accumulator-dtype)
     operations result result-dtype)))
